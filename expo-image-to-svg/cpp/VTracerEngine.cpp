// ═══════════════════════════════════════════════════════════════════════════
//  VTracerEngine.cpp  —  production-grade SVG vectoriser 
// ═══════════════════════════════════════════════════════════════════════════

#include "VTracerEngine.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <time.h>   // clock_gettime

// ── Logging ─────────────────────────────────────────────────────────────────
// On Android link with -llog.  On other platforms falls back to stderr.
#ifdef __ANDROID__
#  include <android/log.h>
#  define VT_LOG(fmt, ...)  __android_log_print(ANDROID_LOG_DEBUG, "VTracerEngine", fmt, ##__VA_ARGS__)
#  define VT_WARN(fmt, ...) __android_log_print(ANDROID_LOG_WARN,  "VTracerEngine", fmt, ##__VA_ARGS__)
#  define VT_ERR(fmt, ...)  __android_log_print(ANDROID_LOG_ERROR, "VTracerEngine", fmt, ##__VA_ARGS__)
#else
#  include <cstdio>
#  define VT_LOG(fmt, ...)  fprintf(stderr, "[VTracer] "  fmt "\n", ##__VA_ARGS__)
#  define VT_WARN(fmt, ...) fprintf(stderr, "[VTracer WARN] " fmt "\n", ##__VA_ARGS__)
#  define VT_ERR(fmt, ...)  fprintf(stderr, "[VTracer ERROR] " fmt "\n", ##__VA_ARGS__)
#endif

// ── Timing helper ────────────────────────────────────────────────────────────
static inline double vt_now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// Optional SIMD — guarded so the file compiles without intrinsics headers
#if defined(__SSE4_2__)
#  include <smmintrin.h>
#  define VT_HAS_SSE 1
#elif defined(__AVX2__)
#  include <immintrin.h>
#  define VT_HAS_AVX 1
#endif

namespace vtracer {

// ───────────────────────────────────────────────────────────────────────────
//  Defaults
// ───────────────────────────────────────────────────────────────────────────
static constexpr int   kDefaultColorPrecision  = 6;
static constexpr float kDefaultCornerThreshold = 120.f;
static constexpr int   kDefaultFilterSpeckle   = 4;
static constexpr float kDefaultRdpEpsilon      = 1.5f;
static constexpr int   kDefaultPathPrecision   = 1;
static constexpr float kDefaultBlurRadius      = 1.0f;
static constexpr int   kMaxPaletteSize         = 256;

// Schneider fitting parameters
static constexpr int   kMaxFitIter             = 8;
static constexpr float kFitTolerance           = 0.5f;

// ───────────────────────────────────────────────────────────────────────────
//  Geometry types
// ───────────────────────────────────────────────────────────────────────────
struct Point   { float x, y; };
struct Segment { bool isCurve; Point cp1, cp2, end; };

static inline Point operator+(const Point& a, const Point& b){ return {a.x+b.x,a.y+b.y}; }
static inline Point operator-(const Point& a, const Point& b){ return {a.x-b.x,a.y-b.y}; }
static inline Point operator*(float s, const Point& p){ return {s*p.x,s*p.y}; }
static inline Point operator*(const Point& p, float s){ return {s*p.x,s*p.y}; }
static inline float dot(const Point& a, const Point& b){ return a.x*b.x+a.y*b.y; }
static inline float lenSq(const Point& p){ return p.x*p.x+p.y*p.y; }

// ───────────────────────────────────────────────────────────────────────────
//  Moore neighbourhood — 8-connected clockwise from North
//   7 0 1 / 6 * 2 / 5 4 3
// ───────────────────────────────────────────────────────────────────────────
static const int DX[8] = { 0,  1,  1,  1,  0, -1, -1, -1 };
static const int DY[8] = {-1, -1,  0,  1,  1,  1,  0, -1 };

// ───────────────────────────────────────────────────────────────────────────
//  Union-Find (path compression + rank)
// ───────────────────────────────────────────────────────────────────────────
struct UnionFind {
    std::vector<int> parent, rank_;
    explicit UnionFind(int n) : parent(n), rank_(n, 0) {
        std::iota(parent.begin(), parent.end(), 0);
    }
    int find(int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }
    void unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return;
        if (rank_[a] < rank_[b]) std::swap(a, b);
        parent[b] = a;
        if (rank_[a] == rank_[b]) ++rank_[a];
    }
};

// ───────────────────────────────────────────────────────────────────────────
//  Colour helpers
// ───────────────────────────────────────────────────────────────────────────
static inline uint32_t packRGB(uint8_t r, uint8_t g, uint8_t b) {
    return ((uint32_t)r << 16) | ((uint32_t)g << 8) | b;
}
static inline uint32_t rgb24(uint32_t c)  { return c & 0x00FFFFFFu; }
static inline uint8_t  rCh(uint32_t c)   { return (c >> 16) & 0xFF; }
static inline uint8_t  gCh(uint32_t c)   { return (c >>  8) & 0xFF; }
static inline uint8_t  bCh(uint32_t c)   { return  c        & 0xFF; }

static float luminance(uint32_t c) {
    return 0.299f*(float)rCh(c) + 0.587f*(float)gCh(c) + 0.114f*(float)bCh(c);
}

// ───────────────────────────────────────────────────────────────────────────
//  [E2] LUT-accelerated CIE-Lab pipeline
// ───────────────────────────────────────────────────────────────────────────
struct LabLUT {
    float linearise[256];
    float labF[1024];
    static constexpr float kFDomain = 1.1f;
    static constexpr int   kFSize   = 1024;

    LabLUT() {
        for (int i = 0; i < 256; ++i) {
            float u = (float)i / 255.f;
            linearise[i] = (u <= 0.04045f)
                ? u / 12.92f
                : std::pow((u + 0.055f) / 1.055f, 2.4f);
        }
        for (int i = 0; i < kFSize; ++i) {
            float t = kFDomain * (float)i / (float)kFSize;
            labF[i] = (t > 0.008856f)
                ? std::cbrt(t)
                : (7.787f * t + 16.f / 116.f);
        }
    }

    inline float f(float t) const {
        int idx = (int)(t * (float)kFSize / kFDomain);
        if (idx < 0) idx = 0;
        if (idx >= kFSize) idx = kFSize - 1;
        return labF[idx];
    }
} g_lut;

struct Lab { float L, a, b; };

static Lab rgbToLabLUT(uint32_t c) {
    float rl = g_lut.linearise[rCh(c)];
    float gl = g_lut.linearise[gCh(c)];
    float bl = g_lut.linearise[bCh(c)];
    float X = rl*0.4124564f + gl*0.3575761f + bl*0.1804375f;
    float Y = rl*0.2126729f + gl*0.7151522f + bl*0.0721750f;
    float Z = rl*0.0193339f + gl*0.1191920f + bl*0.9503041f;
    X /= 0.95047f; Z /= 1.08883f;
    float fx = g_lut.f(X), fy = g_lut.f(Y), fz = g_lut.f(Z);
    return {116.f*fy - 16.f, 500.f*(fx - fy), 200.f*(fy - fz)};
}

static float labDistSq(uint32_t a, uint32_t b) {
    Lab la = rgbToLabLUT(rgb24(a)), lb = rgbToLabLUT(rgb24(b));
    float dL = la.L-lb.L, da = la.a-lb.a, db = la.b-lb.b;
    return dL*dL + da*da + db*db;
}

// ───────────────────────────────────────────────────────────────────────────
//  [E2] Stage 0 — SIMD-friendly Separable Gaussian pre-blur
// ───────────────────────────────────────────────────────────────────────────
static std::vector<uint8_t> gaussianBlur(
    const uint8_t* src, int W, int H, float sigma)
{
    if (sigma < 0.1f)
        return std::vector<uint8_t>(src, src + (size_t)W*H*4);

    sigma = std::min(sigma, 3.f);
    const int radius = (int)std::ceil(3.f * sigma);
    const int ksize  = 2*radius + 1;

    std::vector<float> kern(ksize);
    float ksum = 0.f;
    for (int i = 0; i < ksize; ++i) {
        float x = (float)(i - radius);
        kern[i] = std::exp(-0.5f*x*x/(sigma*sigma));
        ksum   += kern[i];
    }
    for (float& k : kern) k /= ksum;

    const int N = W * H;
    std::vector<float>   tmp(N*4, 0.f);
    std::vector<uint8_t> dst(N*4);

    // Horizontal pass
    std::vector<float> row_scratch((W + 2*radius) * 4);

    for (int y = 0; y < H; ++y) {
        const uint8_t* srcy = src + y*W*4;
        float* pad = row_scratch.data();

        for (int px = -radius; px < W + radius; ++px) {
            int sx = std::clamp(px, 0, W-1);
            const uint8_t* p = srcy + sx*4;
            float* q = pad + (px + radius)*4;
            q[0]=(float)p[0]; q[1]=(float)p[1];
            q[2]=(float)p[2]; q[3]=(float)p[3];
        }

        float* outy = tmp.data() + y*W*4;
        for (int x = 0; x < W; ++x) {
            float r=0,g=0,b=0,a=0;
            const float* base = pad + x*4;
            for (int k = 0; k < ksize; ++k) {
                float w = kern[k];
                const float* q = base + k*4;
                r += w*q[0]; g += w*q[1]; b += w*q[2]; a += w*q[3];
            }
            float* o = outy + x*4;
            o[0]=r; o[1]=g; o[2]=b; o[3]=a;
        }
    }

    // Vertical pass (column-major strip for cache)
    static constexpr int kColStrip = 8;
    std::vector<float> col_scratch((H + 2*radius) * kColStrip * 4);

    for (int x0 = 0; x0 < W; x0 += kColStrip) {
        int x1 = std::min(x0 + kColStrip, W);
        int ncols = x1 - x0;

        for (int py = -radius; py < H + radius; ++py) {
            int sy = std::clamp(py, 0, H-1);
            const float* src_row = tmp.data() + sy*W*4;
            float* dst_row = col_scratch.data() + (py + radius)*ncols*4;
            for (int ci = 0; ci < ncols; ++ci) {
                const float* q = src_row + (x0+ci)*4;
                float* d = dst_row + ci*4;
                d[0]=q[0]; d[1]=q[1]; d[2]=q[2]; d[3]=q[3];
            }
        }

        for (int y = 0; y < H; ++y) {
            for (int ci = 0; ci < ncols; ++ci) {
                float r=0,g=0,b=0,a=0;
                for (int k = 0; k < ksize; ++k) {
                    float w = kern[k];
                    const float* q = col_scratch.data() + (y+k)*ncols*4 + ci*4;
                    r += w*q[0]; g += w*q[1]; b += w*q[2]; a += w*q[3];
                }
                uint8_t* d = dst.data() + (y*W + x0+ci)*4;
                d[0] = (uint8_t)std::clamp((int)(r+.5f),0,255);
                d[1] = (uint8_t)std::clamp((int)(g+.5f),0,255);
                d[2] = (uint8_t)std::clamp((int)(b+.5f),0,255);
                d[3] = (uint8_t)std::clamp((int)(a+.5f),0,255);
            }
        }
    }
    return dst;
}

// ───────────────────────────────────────────────────────────────────────────
//  Stage 1 — Median-cut palette quantisation
// ───────────────────────────────────────────────────────────────────────────
struct ColorEntry { uint32_t color; int count; };

static int medianCutSplit(
    std::vector<ColorEntry>& E, int lo, int hi, int ch)
{
    std::sort(E.begin()+lo, E.begin()+hi,
        [ch](const ColorEntry& a, const ColorEntry& b){
            uint8_t va = ch==0?rCh(a.color):ch==1?gCh(a.color):bCh(a.color);
            uint8_t vb = ch==0?rCh(b.color):ch==1?gCh(b.color):bCh(b.color);
            return va < vb;
        });
    long total=0;
    for (int i=lo;i<hi;++i) total+=E[i].count;
    long half=total/2, acc=0;
    for (int i=lo;i<hi-1;++i) {
        acc += E[i].count;
        if (acc >= half) return i+1;
    }
    return (lo+hi)/2;
}

static uint32_t boxRepresentative(
    const std::vector<ColorEntry>& E, int lo, int hi)
{
    double r=0,g=0,b=0; long total=0;
    for (int i=lo;i<hi;++i){
        double w=E[i].count;
        r+=w*rCh(E[i].color); g+=w*gCh(E[i].color); b+=w*bCh(E[i].color);
        total+=E[i].count;
    }
    if (!total) return 0;
    return packRGB(
        (uint8_t)std::clamp((int)(r/total+.5),0,255),
        (uint8_t)std::clamp((int)(g/total+.5),0,255),
        (uint8_t)std::clamp((int)(b/total+.5),0,255));
}

static int widestChannel(const std::vector<ColorEntry>& E, int lo, int hi) {
    uint8_t rMn=255,rMx=0,gMn=255,gMx=0,bMn=255,bMx=0;
    for (int i=lo;i<hi;++i){
        uint32_t c=E[i].color;
        rMn=std::min(rMn,rCh(c)); rMx=std::max(rMx,rCh(c));
        gMn=std::min(gMn,gCh(c)); gMx=std::max(gMx,gCh(c));
        bMn=std::min(bMn,bCh(c)); bMx=std::max(bMx,bCh(c));
    }
    int rR=rMx-rMn, gR=gMx-gMn, bR=bMx-bMn;
    return (rR>=gR&&rR>=bR)?0:(gR>=bR)?1:2;
}

static std::vector<uint32_t> medianCutPalette(
    std::vector<ColorEntry>& E, int target)
{
    if (E.empty()) return {};
    target = std::clamp(target, 1, kMaxPaletteSize);

    if ((int)E.size() <= target) {
        std::vector<uint32_t> p; p.reserve(E.size());
        for (auto& e:E) p.push_back(rgb24(e.color));
        return p;
    }

    struct Box { int lo, hi; };
    std::vector<Box> boxes;
    boxes.push_back({0,(int)E.size()});

    while ((int)boxes.size() < target) {
        int bestBox=-1, bestRange=-1;
        for (int bi=0;bi<(int)boxes.size();++bi) {
            auto& bx=boxes[bi];
            if (bx.hi-bx.lo<=1) continue;
            int ch=widestChannel(E,bx.lo,bx.hi);
            uint8_t mn=255,mx=0;
            for (int i=bx.lo;i<bx.hi;++i){
                uint8_t v=ch==0?rCh(E[i].color):ch==1?gCh(E[i].color):bCh(E[i].color);
                mn=std::min(mn,v); mx=std::max(mx,v);
            }
            int range=(int)(mx-mn);
            if (range>bestRange){bestRange=range;bestBox=bi;}
        }
        if (bestBox==-1) break;
        auto& bx = boxes[bestBox];
        int ch  = widestChannel(E,bx.lo,bx.hi);
        int mid = medianCutSplit(E,bx.lo,bx.hi,ch);
        Box left={bx.lo,mid}, right={mid,bx.hi};
        boxes[bestBox]=left;
        boxes.push_back(right);
    }

    std::vector<uint32_t> pal; pal.reserve(boxes.size());
    for (auto& bx:boxes)
        if (bx.hi>bx.lo) pal.push_back(boxRepresentative(E,bx.lo,bx.hi));
    return pal;
}

// ───────────────────────────────────────────────────────────────────────────
//  Stage 1+2 — Quantise + perceptual gradient merge
// ───────────────────────────────────────────────────────────────────────────
static std::vector<uint32_t> buildPaletteAndAssign(
    const uint8_t*         pixels,
    int W, int H,
    const Options&         opt,
    std::vector<uint32_t>& pixelColor)
{
    const int N = W*H;
    pixelColor.assign(N, 0xFFFFFFFFu);

    std::unordered_map<uint32_t,int> freq;
    freq.reserve(4096);
    bool anyOpaque = false;

    for (int i=0;i<N;++i){
        const uint8_t* p = pixels+i*4;
        if (p[3]==0) continue;
        anyOpaque = true;
        if (opt.color_mode == ColorMode::BlackAndWhite) {
            float lum = 0.299f*p[0]+0.587f*p[1]+0.114f*p[2];
            freq[lum>=128.f ? 0x00FFFFFFu : 0x00000000u]++;
        } else {
            freq[packRGB(p[0],p[1],p[2])]++;
        }
    }
    if (!anyOpaque) {
        VT_WARN("buildPaletteAndAssign: image has no opaque pixels (all alpha==0). Returning empty SVG.");
        return {};
    }

    const int targetSz = std::min(1 << std::clamp(opt.color_precision,1,8),
                                  kMaxPaletteSize);
    VT_LOG("Stage 1: %d unique colours -> target palette size %d", (int)freq.size(), targetSz);

    std::vector<ColorEntry> entries;
    entries.reserve(freq.size());
    for (auto& [c,cnt]:freq) entries.push_back({rgb24(c),cnt});

    std::vector<uint32_t> palette = medianCutPalette(entries, targetSz);
    if (palette.empty()) {
        VT_WARN("buildPaletteAndAssign: medianCutPalette returned empty palette.");
        return {};
    }
    VT_LOG("Stage 1: palette built with %d entries", (int)palette.size());

    const int K = (int)palette.size();
    std::vector<Lab> palLab(K);
    for (int i=0;i<K;++i) palLab[i] = rgbToLabLUT(palette[i]);

    UnionFind uf(K);
    if (opt.gradient_step > 0.f) {
        const float thresh2 = opt.gradient_step * opt.gradient_step;
        int merges = 0;
        for (int ii=0;ii<K;++ii)
            for (int jj=ii+1;jj<K;++jj){
                const Lab& la=palLab[ii]; const Lab& lb=palLab[jj];
                float dL=la.L-lb.L, da=la.a-lb.a, db=la.b-lb.b;
                if (dL*dL+da*da+db*db <= thresh2) { uf.unite(ii,jj); ++merges; }
            }
        VT_LOG("Stage 2: gradient merge (step=%.2f) performed %d merges", (double)opt.gradient_step, merges);
    }

    struct Acc { double r,g,b; long count; };
    std::unordered_map<int,Acc> groupAcc;
    for (int i=0;i<K;++i){
        uint32_t c   = palette[i];
        long     cnt = freq.count(c) ? freq.at(c) : 1;
        int      root= uf.find(i);
        auto&    acc = groupAcc[root];
        acc.r+=cnt*rCh(c); acc.g+=cnt*gCh(c); acc.b+=cnt*bCh(c);
        acc.count+=cnt;
    }
    std::unordered_map<int,uint32_t> rootColor;
    for (auto& [root,acc]:groupAcc){
        rootColor[root] = packRGB(
            (uint8_t)std::clamp((int)(acc.r/acc.count+.5),0,255),
            (uint8_t)std::clamp((int)(acc.g/acc.count+.5),0,255),
            (uint8_t)std::clamp((int)(acc.b/acc.count+.5),0,255));
    }

    std::unordered_map<uint32_t,uint32_t> nearestCache;
    nearestCache.reserve(freq.size()*2);
    auto nearest = [&](uint32_t c) -> uint32_t {
        auto it = nearestCache.find(c);
        if (it != nearestCache.end()) return it->second;
        Lab lc = rgbToLabLUT(rgb24(c));
        float bestD=1e30f; int bestI=0;
        for (int i=0;i<K;++i){
            const Lab& lp = palLab[i];
            float dL=lc.L-lp.L, da=lc.a-lp.a, db=lc.b-lp.b;
            float d=dL*dL+da*da+db*db;
            if (d<bestD){bestD=d;bestI=i;}
        }
        uint32_t res = rootColor[uf.find(bestI)];
        nearestCache[c] = res;
        return res;
    };

    for (int i=0;i<N;++i){
        const uint8_t* p = pixels+i*4;
        if (p[3]==0) continue;
        uint32_t raw;
        if (opt.color_mode == ColorMode::BlackAndWhite) {
            float lum=0.299f*p[0]+0.587f*p[1]+0.114f*p[2];
            raw = lum>=128.f ? 0x00FFFFFFu : 0x00000000u;
        } else {
            raw = packRGB(p[0],p[1],p[2]);
        }
        pixelColor[i] = nearest(raw);
    }

    std::unordered_map<uint32_t,bool> seen;
    std::vector<uint32_t> used;
    for (int i=0;i<N;++i){
        if (pixelColor[i]==0xFFFFFFFFu) continue;
        if (seen.emplace(pixelColor[i],true).second)
            used.push_back(pixelColor[i]);
    }
    std::sort(used.begin(),used.end(),
              [](uint32_t a,uint32_t b){return luminance(a)<luminance(b);});
    VT_LOG("Stage 1+2: %d distinct colours in use after assignment", (int)used.size());
    return used;
}

// ───────────────────────────────────────────────────────────────────────────
//  Stage 3 — BFS connected-component labelling (4-connectivity)
// ───────────────────────────────────────────────────────────────────────────
static std::vector<int> labelComponents(
    const std::vector<uint32_t>& pixelColor,
    int W, int H,
    std::vector<uint32_t>& componentColor,
    std::vector<int>&      componentSize)
{
    const int N=W*H;
    std::vector<int> label(N,-1);
    componentColor.clear(); componentSize.clear();
    static const int ox[4]={1,-1,0,0}, oy[4]={0,0,1,-1};
    std::vector<int> q; q.reserve(2048);

    for (int i=0;i<N;++i){
        if (pixelColor[i]==0xFFFFFFFFu || label[i]!=-1) continue;
        uint32_t myColor=pixelColor[i];
        int lbl=(int)componentColor.size();
        componentColor.push_back(myColor);
        componentSize.push_back(0);
        label[i]=lbl; q.clear(); q.push_back(i);
        int head=0;
        while (head<(int)q.size()){
            int cur=q[head++];
            ++componentSize[lbl];
            int cx=cur%W, cy=cur/W;
            for (int d=0;d<4;++d){
                int nx=cx+ox[d], ny=cy+oy[d];
                if ((unsigned)nx>=(unsigned)W||(unsigned)ny>=(unsigned)H) continue;
                int ni=ny*W+nx;
                if (label[ni]!=-1||pixelColor[ni]!=myColor) continue;
                label[ni]=lbl; q.push_back(ni);
            }
        }
    }
    return label;
}

// ───────────────────────────────────────────────────────────────────────────
//  Stage 4 — Speckle-filter BFS clear
// ───────────────────────────────────────────────────────────────────────────
static void clearComponent(
    int startIdx, int lbl,
    const std::vector<int>& labelMap,
    std::vector<uint8_t>&   occ,
    int W, int H)
{
    static const int ox[4]={1,-1,0,0}, oy[4]={0,0,1,-1};
    std::vector<int> q; q.reserve(256);
    q.push_back(startIdx); occ[startIdx]=0;
    int head=0;
    while (head<(int)q.size()){
        int cur=q[head++];
        int cx=cur%W, cy=cur/W;
        for (int d=0;d<4;++d){
            int nx=cx+ox[d], ny=cy+oy[d];
            if ((unsigned)nx>=(unsigned)W||(unsigned)ny>=(unsigned)H) continue;
            int ni=ny*W+nx;
            if (occ[ni]==0||labelMap[ni]!=lbl) continue;
            occ[ni]=0; q.push_back(ni);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  [E1] Stage 5 — Shared Topology Management
// ═══════════════════════════════════════════════════════════════════════════
struct EdgePixel {
    int x, y;
    int axis;
    uint32_t cA, cB;
};

struct SharedEdgeGraph {
    std::unordered_map<uint64_t, std::vector<EdgePixel>> edges;

    static uint64_t key(uint32_t a, uint32_t b) {
        if (a > b) std::swap(a, b);
        return ((uint64_t)a << 32) | (uint64_t)b;
    }

    void insert(int x, int y, int axis, uint32_t cA, uint32_t cB) {
        edges[key(cA,cB)].push_back({x, y, axis, cA, cB});
    }

    bool hasEdge(uint32_t cA, uint32_t cB) const {
        return edges.count(key(cA,cB)) > 0;
    }
};

static SharedEdgeGraph buildEdgeGraph(
    const std::vector<uint32_t>& pixelColor, int W, int H)
{
    SharedEdgeGraph graph;
    graph.edges.reserve(256);

    for (int y = 0; y < H-1; ++y) {
        for (int x = 0; x < W; ++x) {
            uint32_t cT = pixelColor[y*W+x];
            uint32_t cB = pixelColor[(y+1)*W+x];
            if (cT == 0xFFFFFFFFu || cB == 0xFFFFFFFFu) continue;
            if (cT != cB) graph.insert(x, y, 0, cT, cB);
        }
    }
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W-1; ++x) {
            uint32_t cL = pixelColor[y*W+x];
            uint32_t cR = pixelColor[y*W+x+1];
            if (cL == 0xFFFFFFFFu || cR == 0xFFFFFFFFu) continue;
            if (cL != cR) graph.insert(x, y, 1, cL, cR);
        }
    }
    return graph;
}

// ───────────────────────────────────────────────────────────────────────────
//  Stage 6 — Moore boundary trace (Jacob's stopping criterion)
//
//  [FIX-1] Only follow occ==1 (unvisited foreground) pixels as neighbours.
//           The original used !occ[...] which is false for 0 but TRUE for
//           both 1 and 2 (already-traced). Following occ==2 pixels lets the
//           tracer re-enter already-processed boundary rings, defeating the
//           Jacob stopping criterion and causing the loop to run the full
//           maxSteps guard on every non-trivial shape.
//
//  [FIX-2] maxSteps is now passed in as componentMaxSteps, capped at
//           min(componentPixels*8 + 16, W*H + 8).  The original used W*H*2+8
//           unconditionally — that is the right cap for the *entire image*
//           but wildly over-generous per component, causing multi-second
//           stalls when many components are present.
// ───────────────────────────────────────────────────────────────────────────
static std::vector<Point> traceBoundary(
    int startX, int startY, int W, int H,
    std::vector<uint8_t>& occ,
    int componentMaxSteps)   // [FIX-2] caller supplies per-component cap
{
    std::vector<Point> path; path.reserve(64);

    int entryDir=-1;
    for (int d=0;d<8;++d){
        int nx=startX+DX[d], ny=startY+DY[d];
        if ((unsigned)nx>=(unsigned)W||(unsigned)ny>=(unsigned)H) continue;
        // [FIX-1] only unvisited foreground pixels qualify as entry context
        if (occ[ny*W+nx] == 1){entryDir=d;break;}
    }
    if (entryDir==-1){
        // Isolated pixel (no foreground neighbour in any direction)
        path.push_back({startX+.5f,startY+.5f});
        return path;
    }

    int curX=startX, curY=startY;
    int fromDir=(entryDir+5)%8;
    int firstEntryDir=-1;
    bool firstVisit=true;

    for (int step=0; step < componentMaxSteps; ++step){
        path.push_back({curX+.5f,curY+.5f});
        occ[curY*W+curX]=2;
        bool found=false;
        for (int i=0;i<8;++i){
            int dir=(fromDir+i)%8;
            int nx=curX+DX[dir], ny=curY+DY[dir];
            if ((unsigned)nx>=(unsigned)W||(unsigned)ny>=(unsigned)H) continue;
            // [FIX-1] only follow unvisited foreground — never re-enter occ==2
            if (occ[ny*W+nx] != 1) continue;
            if (nx==startX&&ny==startY){
                if (firstVisit){firstEntryDir=dir;firstVisit=false;}
                else if (dir==firstEntryDir) return path;
            } else firstVisit=false;
            curX=nx; curY=ny;
            fromDir=(dir+5)%8;
            found=true; break;
        }
        if (!found) break;

        // [FIX-2] Emit a warning if we are about to exhaust the step budget
        if (step == componentMaxSteps - 2) {
            VT_WARN("traceBoundary: step budget nearly exhausted "
                    "(componentMaxSteps=%d, path.size=%d, start=(%d,%d)). "
                    "Possible degenerate shape — trace will be truncated.",
                    componentMaxSteps, (int)path.size(), startX, startY);
        }
    }
    return path;
}

// ───────────────────────────────────────────────────────────────────────────
//  Snap boundary vertices to exact shared-edge grid lines  [E1]
// ───────────────────────────────────────────────────────────────────────────
static void snapToSharedEdges(
    std::vector<Point>& path,
    const std::vector<uint32_t>& pixelColor,
    uint32_t myColor,
    int W, int H,
    float snapDist = 0.6f)
{
    for (auto& pt : path) {
        int px = std::clamp((int)pt.x, 0, W-1);
        int py = std::clamp((int)pt.y, 0, H-1);

        float nearestX = std::round(pt.x);
        if (std::abs(pt.x - nearestX) < snapDist) {
            int nx = (int)nearestX;
            if (nx > 0 && nx < W) {
                int iL = py*W + (nx-1), iR = py*W + nx;
                uint32_t cL = (iL>=0&&iL<W*H) ? pixelColor[iL] : 0xFFFFFFFFu;
                uint32_t cR = (iR>=0&&iR<W*H) ? pixelColor[iR] : 0xFFFFFFFFu;
                if ((cL==myColor) != (cR==myColor))
                    pt.x = (float)nx;
            }
        }
        float nearestY = std::round(pt.y);
        if (std::abs(pt.y - nearestY) < snapDist) {
            int ny = (int)nearestY;
            if (ny > 0 && ny < H) {
                int iT = (ny-1)*W + px, iB = ny*W + px;
                uint32_t cT = (iT>=0&&iT<W*H) ? pixelColor[iT] : 0xFFFFFFFFu;
                uint32_t cB = (iB>=0&&iB<W*H) ? pixelColor[iB] : 0xFFFFFFFFu;
                if ((cT==myColor) != (cB==myColor))
                    pt.y = (float)ny;
            }
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
//  Stage 7 — Ramer-Douglas-Peucker simplification (iterative, stack-based)
// ───────────────────────────────────────────────────────────────────────────
static float ptSegDistSq(const Point& p, const Point& a, const Point& b) {
    float dx=b.x-a.x, dy=b.y-a.y;
    float lenSq_=dx*dx+dy*dy;
    if (lenSq_<1e-12f){float ex=p.x-a.x,ey=p.y-a.y;return ex*ex+ey*ey;}
    float t=std::clamp(((p.x-a.x)*dx+(p.y-a.y)*dy)/lenSq_,0.f,1.f);
    float qx=a.x+t*dx-p.x, qy=a.y+t*dy-p.y;
    return qx*qx+qy*qy;
}

static std::vector<Point> rdpSimplify(const std::vector<Point>& pts, float eps) {
    const int n=(int)pts.size();
    if (n<=2) return pts;
    const float epsSq=eps*eps;
    std::vector<bool> keep(n,false);
    keep[0]=keep[n-1]=true;

    struct Frame{int lo,hi;};
    std::vector<Frame> stack;
    stack.push_back({0,n-1});
    while (!stack.empty()){
        auto [a,b]=stack.back(); stack.pop_back();
        if (b-a<=1) continue;
        float maxD=0.f; int maxI=a;
        for (int i=a+1;i<b;++i){
            float d=ptSegDistSq(pts[i],pts[a],pts[b]);
            if (d>maxD){maxD=d;maxI=i;}
        }
        if (maxD>epsSq){
            keep[maxI]=true;
            stack.push_back({a,maxI});
            stack.push_back({maxI,b});
        }
    }
    std::vector<Point> out; out.reserve(n);
    for (int i=0;i<n;++i) if (keep[i]) out.push_back(pts[i]);
    return out;
}

// ───────────────────────────────────────────────────────────────────────────
//  Stage 8 — Corner detection
// ───────────────────────────────────────────────────────────────────────────
static std::vector<uint8_t> detectCorners(
    const std::vector<Point>& pts, float thresh_deg)
{
    const int n=(int)pts.size();
    std::vector<uint8_t> c(n,0);
    if (n<3){std::fill(c.begin(),c.end(),1);return c;}
    const float thresh_rad=thresh_deg*(float)M_PI/180.f;
    for (int i=0;i<n;++i){
        const Point& prev=pts[(i+n-1)%n];
        const Point& cur =pts[i];
        const Point& next=pts[(i+1)%n];
        float ax=cur.x-prev.x, ay=cur.y-prev.y;
        float bx=next.x-cur.x, by=next.y-cur.y;
        float lA=std::sqrt(ax*ax+ay*ay);
        float lB=std::sqrt(bx*bx+by*by);
        if (lA<1e-6f||lB<1e-6f){c[i]=1;continue;}
        float d=std::clamp((ax*bx+ay*by)/(lA*lB),-1.f,1.f);
        c[i]=(std::acos(d)<thresh_rad)?1:0;
    }
    return c;
}

// ═══════════════════════════════════════════════════════════════════════════
//  [E3] Stage 9 — Constrained Least-Squares Bézier Fitting
// ═══════════════════════════════════════════════════════════════════════════

// Evaluate cubic Bézier at t
static Point bezier(const Point& P0, const Point& P1,
                    const Point& P2, const Point& P3, float t)
{
    float mt=1.f-t;
    float b0=mt*mt*mt, b1=3.f*mt*mt*t, b2=3.f*mt*t*t, b3=t*t*t;
    return {b0*P0.x+b1*P1.x+b2*P2.x+b3*P3.x,
            b0*P0.y+b1*P1.y+b2*P2.y+b3*P3.y};
}

// Chord-length parameterise a range of pts[lo..hi)
static std::vector<float> chordParam(
    const std::vector<Point>& pts, int lo, int hi)
{
    int n=hi-lo;
    std::vector<float> u(n, 0.f);
    for (int i=1;i<n;++i){
        float dx=pts[lo+i].x-pts[lo+i-1].x;
        float dy=pts[lo+i].y-pts[lo+i-1].y;
        u[i]=u[i-1]+std::sqrt(dx*dx+dy*dy);
    }
    float tot=u[n-1];
    if (tot>1e-6f) for (float& v:u) v/=tot;
    return u;
}

struct CubicFit { Point P1, P2; float maxResidSq; };

// [FIX-4] Accept an external parameterisation `uExt` so the Newton-refined
//         `u` from the previous iteration is actually used rather than
//         recomputed from scratch each time.
static CubicFit fitCubicSegment(
    const std::vector<Point>& raw, int lo, int hi,
    const Point& P0, const Point& P3,
    const std::vector<float>* uExt = nullptr)   // [FIX-4]
{
    const int n = hi - lo;
    if (n <= 2) {
        Point P1 = {P0.x + (P3.x-P0.x)/3.f, P0.y + (P3.y-P0.y)/3.f};
        Point P2 = {P0.x + 2.f*(P3.x-P0.x)/3.f, P0.y + 2.f*(P3.y-P0.y)/3.f};
        return {P1, P2, 0.f};
    }

    // [FIX-4] Use caller-supplied u if provided, otherwise compute from chord length
    std::vector<float> u_storage;
    const std::vector<float>* u = uExt;
    if (!u || (int)u->size() != n) {
        u_storage = chordParam(raw, lo, hi);
        u = &u_storage;
    }

    double A00=0,A01=0,A11=0;
    double Bx0=0,By0=0,Bx1=0,By1=0;

    for (int i=0;i<n;++i){
        float t=(*u)[i];
        float mt=1.f-t;
        float c0=mt*mt*mt, c1=3.f*mt*mt*t, c2=3.f*mt*t*t, c3=t*t*t;
        float Qx=raw[lo+i].x - c0*P0.x - c3*P3.x;
        float Qy=raw[lo+i].y - c0*P0.y - c3*P3.y;
        A00+=c1*c1; A01+=c1*c2; A11+=c2*c2;
        Bx0+=c1*Qx; By0+=c1*Qy;
        Bx1+=c2*Qx; By1+=c2*Qy;
    }

    double det = A00*A11 - A01*A01;
    Point P1, P2;
    if (std::abs(det) < 1e-10) {
        // Singular system — fall back to linear interpolation of control points
        P1 = {P0.x + (P3.x-P0.x)/3.f, P0.y + (P3.y-P0.y)/3.f};
        P2 = {P0.x + 2.f*(P3.x-P0.x)/3.f, P0.y + 2.f*(P3.y-P0.y)/3.f};
    } else {
        double invDet = 1.0 / det;
        P1.x = (float)((A11*Bx0 - A01*Bx1)*invDet);
        P1.y = (float)((A11*By0 - A01*By1)*invDet);
        P2.x = (float)((A00*Bx1 - A01*Bx0)*invDet);
        P2.y = (float)((A00*By1 - A01*By0)*invDet);
    }

    float maxRes = 0.f;
    for (int i=0;i<n;++i){
        Point b = bezier(P0, P1, P2, P3, (*u)[i]);
        float dx=b.x-raw[lo+i].x, dy=b.y-raw[lo+i].y;
        float r=dx*dx+dy*dy;
        if (r>maxRes) maxRes=r;
    }
    return {P1, P2, maxRes};
}

// Newton reparameterisation
static std::vector<float> reparameterise(
    const std::vector<Point>& raw, int lo, int hi,
    const Point& P0, const Point& P1,
    const Point& P2, const Point& P3,
    const std::vector<float>& u)
{
    int n = hi - lo;
    std::vector<float> u2(n);
    for (int i=0;i<n;++i){
        float t = u[i];
        Point Bt  = bezier(P0,P1,P2,P3,t);
        float mt=1.f-t;
        Point d0 = P1 - P0, d1 = P2 - P1, d2 = P3 - P2;
        Point Bp = {3.f*(mt*mt*d0.x + 2.f*mt*t*d1.x + t*t*d2.x),
                    3.f*(mt*mt*d0.y + 2.f*mt*t*d1.y + t*t*d2.y)};
        Point diff = {Bt.x - raw[lo+i].x, Bt.y - raw[lo+i].y};
        float denom = dot(Bp,Bp);
        float delta = (denom > 1e-8f) ? dot(diff, Bp) / denom : 0.f;
        u2[i] = std::clamp(t - delta, 0.f, 1.f);
    }
    return u2;
}

// Recursive subdivision Bézier fitter
// [FIX-4] Pass the refined `u` into fitCubicSegment each iteration instead
//         of letting it recompute chordParam from scratch and discard the
//         Newton step.
static void fitBezierRecursive(
    const std::vector<Point>& raw, int lo, int hi,
    const Point& P0, const Point& P3,
    std::vector<Segment>& out,
    int depth = 0,
    float fit_tolerance)
{
    static constexpr int kMaxDepth = 6;
    if (hi - lo <= 1 || depth > kMaxDepth) {
        out.push_back({false,{},{},P3});
        return;
    }

    // [FIX-4] Maintain u across iterations so Newton refinement accumulates
    std::vector<float> u = chordParam(raw, lo, hi);

    Point P1_best, P2_best;
    float bestRes = 1e30f;
    for (int iter=0; iter<kMaxFitIter; ++iter){
        // [FIX-4] Pass current u into fitter so the refined parameterisation is used
        CubicFit fit = fitCubicSegment(raw, lo, hi, P0, P3, &u);
        if (fit.maxResidSq < bestRes){
            bestRes = fit.maxResidSq;
            P1_best = fit.P1; P2_best = fit.P2;
        }
        if (fit.maxResidSq <= fit_tolerance*fit_tolerance) break;
        // Newton reparameterisation — update u for next iteration [FIX-4]
        u = reparameterise(raw, lo, hi, P0, fit.P1, fit.P2, P3, u);
    }

    if (bestRes <= fit_tolerance*fit_tolerance) {
        out.push_back({true, P1_best, P2_best, P3});
        return;
    }

    // Subdivide at max-residual point
    // Reuse the last refined u to find the worst point
    CubicFit fit = fitCubicSegment(raw, lo, hi, P0, P3, &u);
    int worstI = lo;
    {
        float wRes = 0.f;
        for (int i=0; i<hi-lo; ++i){
            Point b = bezier(P0, fit.P1, fit.P2, P3, u[i]);
            float dx=b.x-raw[lo+i].x, dy=b.y-raw[lo+i].y;
            float r=dx*dx+dy*dy;
            if (r>wRes){wRes=r;worstI=lo+i;}
        }
    }
    if (worstI == lo || worstI == hi-1) {
        out.push_back({true, P1_best, P2_best, P3});
        return;
    }
    const Point& Pmid = raw[worstI];
    fitBezierRecursive(raw, lo,    worstI+1, P0,   Pmid, out, depth+1,fit_tolerance);
    fitBezierRecursive(raw, worstI, hi,      Pmid, P3,   out, depth+1,fit_tolerance);
}

// Build the spline for one boundary region
//
// [FIX-3] key→raw mapping is now precomputed in a single O(nk*nr) pass.
//         The original called closestRaw() (O(nr)) inside a double loop
//         (O(nc * nk)), yielding O(nk² * nr) per colour region.
//
// [FIX-5] from==to (single corner) is detected up front and treated as a
//         full-circle run rather than a 1-element degenerate that silently
//         produces no output.
static std::vector<Segment> buildSplineLSQ(
    const std::vector<Point>& keyPts,
    const std::vector<uint8_t>& isCorner,
    const std::vector<Point>& rawPts,
    float fFit_Tolerance)
{
    const int nk = (int)keyPts.size();
    const int nr = (int)rawPts.size();
    if (nk < 2 || nr < 2) return {};

    std::vector<Segment> segs;
    segs.reserve(nk);

    // [FIX-3] Precompute closest raw index for every key vertex in one pass
    std::vector<int> keyToRaw(nk);
    for (int ki = 0; ki < nk; ++ki) {
        const Point& p = keyPts[ki];
        float best = 1e30f; int bi = 0;
        for (int i = 0; i < nr; ++i){
            float dx=rawPts[i].x-p.x, dy=rawPts[i].y-p.y;
            float d=dx*dx+dy*dy;
            if (d<best){best=d;bi=i;}
        }
        keyToRaw[ki] = bi;
    }

    std::vector<int> corners;
    for (int i=0;i<nk;++i) if (isCorner[i]) corners.push_back(i);
    if (corners.empty()) corners.push_back(0);

    const int nc = (int)corners.size();
    auto nxt = [&](int i){ return (i+1)%nk; };

    for (int ci=0; ci<nc; ++ci){
        int from = corners[ci];
        int to   = corners[(ci+1)%nc];

        // [FIX-5] Handle the single-corner (full-circle) case
        bool fullCircle = (from == to);

        std::vector<int> run;
        if (fullCircle) {
            // Walk the entire key-point ring
            for (int k = from;;){
                run.push_back(k);
                k = nxt(k);
                if (k == from) { run.push_back(from); break; }
                if ((int)run.size() > nk + 2) break;
            }
        } else {
            for (int k=from;;k=nxt(k)){
                run.push_back(k);
                if (k==to) break;
                if ((int)run.size()>nk+2) break;
            }
        }
        const int rn = (int)run.size();
        if (rn < 2) continue;

        for (int ri=0; ri<rn-1; ++ri){
            const Point& P0 = keyPts[run[ri]];
            const Point& P3 = keyPts[run[ri+1]];

            // [FIX-3] Use precomputed raw indices
            int rlo = keyToRaw[run[ri]];
            int rhi = keyToRaw[run[ri+1]];

            std::vector<Point> seg_raw;
            if (rlo <= rhi) {
                seg_raw.assign(rawPts.begin()+rlo, rawPts.begin()+rhi+1);
            } else {
                seg_raw.assign(rawPts.begin()+rlo, rawPts.end());
                seg_raw.insert(seg_raw.end(), rawPts.begin(), rawPts.begin()+rhi+1);
            }

            if ((int)seg_raw.size() < 2) {
                segs.push_back({false,{},{},P3});
                continue;
            }

            float dx=P3.x-P0.x, dy=P3.y-P0.y;
            if (dx*dx+dy*dy < 1.f){
                segs.push_back({false,{},{},P3});
                continue;
            }

            fitBezierRecursive(seg_raw, 0, (int)seg_raw.size(), P0, P3, segs,fFit_Tolerance);
        }
    }
    return segs;
}

// ───────────────────────────────────────────────────────────────────────────
//  Stage 10 — Winding-order normalisation
// ───────────────────────────────────────────────────────────────────────────
static float signedArea(const std::vector<Point>& pts){
    float area=0.f; const int n=(int)pts.size();
    for (int i=0;i<n;++i){
        const Point& a=pts[i]; const Point& b=pts[(i+1)%n];
        area+=(a.x*b.y-b.x*a.y);
    }
    return area*0.5f;
}
static void ensureCW (std::vector<Point>& p){if(signedArea(p)<0.f)std::reverse(p.begin(),p.end());}
static void ensureCCW(std::vector<Point>& p){if(signedArea(p)>0.f)std::reverse(p.begin(),p.end());}

// ───────────────────────────────────────────────────────────────────────────
//  Stage 11 — SVG output helpers
// ───────────────────────────────────────────────────────────────────────────
static void appendFloat(std::string& s, float v, int dp){
    char buf[32];
    int len=snprintf(buf,sizeof(buf),"%.*f",dp,(double)v);
    if (dp>0){
        while (len>1&&buf[len-1]=='0') --len;
        if (len>1&&buf[len-1]=='.') --len;
    }
    s.append(buf,len);
}

static void appendSegmentRel(
    std::string& d, const Segment& seg, const Point& prev, int dp)
{
    if (seg.isCurve){
        d+='c';
        appendFloat(d,seg.cp1.x-prev.x,dp); d+=' ';
        appendFloat(d,seg.cp1.y-prev.y,dp); d+=' ';
        appendFloat(d,seg.cp2.x-prev.x,dp); d+=' ';
        appendFloat(d,seg.cp2.y-prev.y,dp); d+=' ';
        appendFloat(d,seg.end.x-prev.x,dp); d+=' ';
        appendFloat(d,seg.end.y-prev.y,dp);
    } else {
        float dx=seg.end.x-prev.x, dy=seg.end.y-prev.y;
        if (std::abs(dy)<1e-4f&&std::abs(dx)>=1e-4f){d+='h';appendFloat(d,dx,dp);}
        else if (std::abs(dx)<1e-4f&&std::abs(dy)>=1e-4f){d+='v';appendFloat(d,dy,dp);}
        else {d+='l';appendFloat(d,dx,dp);d+=' ';appendFloat(d,dy,dp);}
    }
}

static void appendColorHex(std::string& s, uint32_t c){
    c=rgb24(c);
    uint8_t r=rCh(c),g=gCh(c),b=bCh(c);
    char buf[8];
    if ((r&0x0F)==(r>>4)&&(g&0x0F)==(g>>4)&&(b&0x0F)==(b>>4))
        snprintf(buf,sizeof(buf),"#%x%x%x",r>>4,g>>4,b>>4);
    else
        snprintf(buf,sizeof(buf),"#%02x%02x%02x",r,g,b);
    s+=buf;
}

// ───────────────────────────────────────────────────────────────────────────
//  Public entry point
// ───────────────────────────────────────────────────────────────────────────
std::string vectorize(const uint8_t* pixels, int width, int height, Options options)
{
    if (!pixels || width <= 0 || height <= 0) {
        VT_ERR("vectorize: invalid arguments (pixels=%p, width=%d, height=%d)",
               (void*)pixels, width, height);
        assert(false && "vectorize: null pixels or non-positive dimensions");
        return "";
    }

    const double t0 = vt_now_ms();
    VT_LOG("vectorize: start %dx%d (%d pixels)", width, height, width*height);

    // Apply defaults for unset options
    if (options.color_precision  <= 0) options.color_precision  = kDefaultColorPrecision;
    if (options.corner_threshold <= 0) options.corner_threshold = kDefaultCornerThreshold;
    if (options.filter_speckle   <= 0) options.filter_speckle   = kDefaultFilterSpeckle;
    if (options.path_precision   <  0) options.path_precision   = kDefaultPathPrecision;
    if (options.rdp_epsilon      <= 0) options.rdp_epsilon      = kDefaultRdpEpsilon;
    if (options.blur_radius      <  0) options.blur_radius      = kDefaultBlurRadius;
    if (options.fit_tolerance    <= 0) options.fit_tolerance    = kFitTolerance;

    VT_LOG("vectorize: options — color_precision=%d gradient_step=%.2f "
           "color_mode=%s filter_speckle=%d rdp_epsilon=%.2f "
           "corner_threshold=%.1f fit_tolerance=%.2f "
           "blur_radius=%.2f path_precision=%d",
           options.color_precision,
           (double)options.gradient_step,
           options.color_mode == ColorMode::BlackAndWhite ? "BlackAndWhite" : "Color",
           options.filter_speckle,
           (double)options.rdp_epsilon,
           (double)options.corner_threshold,
           (double)options.fit_tolerance,
           (double)options.blur_radius,
           options.path_precision);

    const int dp = std::clamp(options.path_precision, 0, 6);
    const int N  = width * height;

    // Stage 0: Gaussian pre-blur
    double ts = vt_now_ms();
    std::vector<uint8_t> blurred = gaussianBlur(pixels, width, height, options.blur_radius);
    VT_LOG("Stage 0 (blur sigma=%.2f): %.1f ms", (double)options.blur_radius, vt_now_ms()-ts);
    const uint8_t* src = blurred.data();

    // Stages 1+2: quantise + Lab gradient merge
    ts = vt_now_ms();
    std::vector<uint32_t> pixelColor;
    std::vector<uint32_t> palette =
        buildPaletteAndAssign(src, width, height, options, pixelColor);
    VT_LOG("Stage 1+2 (quantise+merge): %.1f ms, palette=%d colours",
           vt_now_ms()-ts, (int)palette.size());

    if (palette.empty()){
        char buf[128];
        snprintf(buf,sizeof(buf),
                 "<svg xmlns='http://www.w3.org/2000/svg' "
                 "viewBox='0 0 %d %d'></svg>",width,height);
        VT_WARN("vectorize: palette empty — returning empty SVG");
        return buf;
    }

    // Stage 3: connected components
    ts = vt_now_ms();
    std::vector<uint32_t> componentColor;
    std::vector<int>      componentSize;
    std::vector<int> labelMap =
        labelComponents(pixelColor, width, height, componentColor, componentSize);
    VT_LOG("Stage 3 (BFS components): %.1f ms, %d components",
           vt_now_ms()-ts, (int)componentColor.size());

    std::unordered_map<uint32_t,std::vector<int>> colorToComponents;
    colorToComponents.reserve(palette.size()*2);
    for (int lbl=0;lbl<(int)componentColor.size();++lbl)
        colorToComponents[componentColor[lbl]].push_back(lbl);

    // [E1] Stage 5: Build shared edge graph
    ts = vt_now_ms();
    SharedEdgeGraph edgeGraph = buildEdgeGraph(pixelColor, width, height);
    VT_LOG("Stage 5 (edge graph): %.1f ms, %d edge groups",
           vt_now_ms()-ts, (int)edgeGraph.edges.size());

    // SVG header
    std::string svg; svg.reserve((size_t)N*6);
    {
        char hdr[256];
        snprintf(hdr,sizeof(hdr),
                 "<svg xmlns='http://www.w3.org/2000/svg' "
                 "viewBox='0 0 %d %d'>",width,height);
        svg+=hdr;
    }

    std::vector<uint8_t> occ(N,0);

    int totalPaths = 0, totalSpeckles = 0, totalTracerMaxStepHits = 0;
    double timeTrace=0, timeRDP=0, timeBezier=0, timeSVG=0;

    // Stages 4–11: per-colour processing
    for (uint32_t color : palette){
        if (!colorToComponents.count(color)) continue;

        std::fill(occ.begin(),occ.end(),0);
        for (int i=0;i<N;++i)
            if (pixelColor[i]==color) occ[i]=1;

        struct PathRecord {
            std::vector<Point>   rawPts;
            std::vector<Point>   pts;
            std::vector<Segment> segs;
            bool                 isHole;
        };
        std::vector<PathRecord> paths;

        for (int y=0;y<height;++y){
            for (int x=0;x<width;++x){
                int idx=y*width+x;
                if (occ[idx]!=1) continue;
                int lbl=labelMap[idx];

                // Stage 4: speckle filter
                if (componentSize[lbl]<options.filter_speckle){
                    clearComponent(idx,lbl,labelMap,occ,width,height);
                    ++totalSpeckles;
                    continue;
                }

                // [FIX-2] Per-component step budget: 8 steps per pixel + slack
                int compMaxSteps = std::min(
                    componentSize[lbl] * 8 + 16,
                    N + 8);

                // Stage 6: boundary trace
                double tTrace = vt_now_ms();
                std::vector<Point> raw=traceBoundary(x,y,width,height,occ,compMaxSteps);
                timeTrace += vt_now_ms()-tTrace;

                // Detect if tracer hit the step budget (truncated trace)
                if ((int)raw.size() >= compMaxSteps - 1) {
                    VT_WARN("Stage 6: tracer hit step budget for component lbl=%d "
                            "size=%d at (%d,%d) — path may be truncated",
                            lbl, componentSize[lbl], x, y);
                    ++totalTracerMaxStepHits;
                }

                clearComponent(idx,lbl,labelMap,occ,width,height);
                if ((int)raw.size()<3) continue;

                // [E1] Snap raw boundary vertices to shared edges
                snapToSharedEdges(raw, pixelColor, color, width, height);

                // Stage 7: RDP simplification
                double tRDP = vt_now_ms();
                std::vector<Point> simplified=rdpSimplify(raw,options.rdp_epsilon);
                timeRDP += vt_now_ms()-tRDP;
                if ((int)simplified.size()<3) continue;

                // Stage 8: corner detection
                std::vector<uint8_t> corners=
                    detectCorners(simplified,options.corner_threshold);

                // [E3] Stage 9: Constrained LS Bézier fit
                double tBez = vt_now_ms();
                std::vector<Segment> segs=buildSplineLSQ(simplified,corners,raw,options.fit_tolerance);
                timeBezier += vt_now_ms()-tBez;
                if (segs.empty()) continue;

                paths.push_back({std::move(raw),std::move(simplified),
                                 std::move(segs),false});
            }
        }
        if (paths.empty()) continue;

        // Stage 10: hole detection + winding order
        struct BBox{float x0,y0,x1,y1;};
        auto getBBox=[](const std::vector<Point>& p)->BBox{
            BBox bb{1e30f,1e30f,-1e30f,-1e30f};
            for (auto& v:p){
                bb.x0=std::min(bb.x0,v.x); bb.y0=std::min(bb.y0,v.y);
                bb.x1=std::max(bb.x1,v.x); bb.y1=std::max(bb.y1,v.y);
            }
            return bb;
        };
        auto bbContains=[](const BBox& o,const BBox& i)->bool{
            return i.x0>=o.x0&&i.y0>=o.y0&&i.x1<=o.x1&&i.y1<=o.y1;
        };

        std::vector<BBox> bboxes;
        bboxes.reserve(paths.size());
        for (auto& pr:paths) bboxes.push_back(getBBox(pr.pts));

        std::vector<int> order((int)paths.size());
        std::iota(order.begin(),order.end(),0);
        std::sort(order.begin(),order.end(),[&](int a,int b){
            float aA=(bboxes[a].x1-bboxes[a].x0)*(bboxes[a].y1-bboxes[a].y0);
            float bA=(bboxes[b].x1-bboxes[b].x0)*(bboxes[b].y1-bboxes[b].y0);
            return aA>bA;
        });
        for (int ii=1;ii<(int)order.size();++ii){
            int i=order[ii];
            for (int jj=0;jj<ii;++jj){
                int j=order[jj];
                if (bbContains(bboxes[j],bboxes[i])){
                    paths[i].isHole=true; break;
                }
            }
        }
        for (auto& pr:paths){
            if (pr.isHole) ensureCCW(pr.pts);
            else           ensureCW(pr.pts);
            auto c2=detectCorners(pr.pts,options.corner_threshold);
            pr.segs=buildSplineLSQ(pr.pts,c2,pr.rawPts,options.fit_tolerance);
        }

        // Stage 11: emit SVG
        double tSVG = vt_now_ms();
        svg+="<g fill='";
        appendColorHex(svg,color);
        svg+="'>";

        for (auto& pr:paths){
            if (pr.segs.empty()) continue;
            std::string d; d.reserve(pr.segs.size()*14);
            const Point& start=pr.pts[0];
            d+='M';
            appendFloat(d,start.x,dp); d+=' ';
            appendFloat(d,start.y,dp);
            Point prev=start;
            for (const Segment& seg:pr.segs){
                appendSegmentRel(d,seg,prev,dp);
                prev=seg.end;
            }
            d+='Z';
            svg+="<path d='"; svg+=d; svg+="'/>";
            ++totalPaths;
        }
        svg+="</g>";
        timeSVG += vt_now_ms()-tSVG;
    }

    svg+="</svg>";

    const double totalMs = vt_now_ms() - t0;
    VT_LOG("vectorize: DONE in %.1f ms | paths=%d speckles_dropped=%d "
           "tracer_budget_hits=%d | "
           "trace=%.1fms rdp=%.1fms bezier=%.1fms svg=%.1fms | "
           "svg_bytes=%zu",
           totalMs, totalPaths, totalSpeckles, totalTracerMaxStepHits,
           timeTrace, timeRDP, timeBezier, timeSVG,
           svg.size());

    if (totalTracerMaxStepHits > 0) {
        VT_WARN("vectorize: %d component(s) hit the tracer step budget. "
                "Consider reducing image resolution or increasing filter_speckle.",
                totalTracerMaxStepHits);
    }

    return svg;
}

} // namespace vtracer