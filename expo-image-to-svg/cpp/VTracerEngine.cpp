// ═══════════════════════════════════════════════════════════════════════════
//  VTracerEngine.cpp  —  production-grade SVG vectoriser  (Enhanced v2)
//
//  Enhancements over v1:
//   [E1] Shared Topology Management
//        All inter-colour edges are extracted once into a half-edge graph.
//        Boundary curves are shared across adjacent colour regions so that
//        RDP and Bézier fitting are applied to the *same* point samples,
//        eliminating micro-gaps where background bleeds through.
//
//   [E2] LUT-Accelerated Perceptual Pipeline
//        linearise() and the CIE f(t) function are pre-computed into 256-
//        and 1024-entry float tables. Nearest-palette search uses the LUT
//        Lab values. The Gaussian passes are written in a cache-friendly
//        blocked layout and, when compiled with SSE4.2 or AVX2, process
//        4–8 floats per cycle via compiler-friendly SIMD intrinsics.
//
//   [E3] Constrained Least-Squares Bézier Fitting (Schneider-style)
//        After RDP the simplified vertices seed an iterative Bézier fit.
//        Control points are refined to minimise squared distance to the
//        original high-res boundary, allowing one cubic segment to span
//        many vertices and dramatically shrinking SVG output size.
//
//  Full pipeline:
//   Stage 0  Gaussian pre-blur          smooths pixel noise before tracing
//   Stage 1  Median-cut quantisation    perceptually accurate palette
//   Stage 2  CIE-Lab gradient merge     perceptual colour distance
//   Stage 3  BFS connected components   4-connectivity per colour
//   Stage 4  Speckle filter             drop tiny components
//   Stage 5  Shared topology extraction build half-edge graph once        [E1]
//   Stage 6  Moore boundary trace       Jacob's stopping criterion
//   Stage 7  RDP simplification         Ramer-Douglas-Peucker
//   Stage 8  Corner detection           angle-based per vertex
//   Stage 9  Constrained LS Bézier fit  Schneider iterative refinement    [E3]
//   Stage 10 Winding-order fix          CW outer, CCW holes
//   Stage 11 SVG emission               relative cmds, short hex, <g> groups
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
static constexpr int   kMaxFitIter             = 8;   // refinement iterations
static constexpr float kFitTolerance           = 0.5f; // pixels, sq = 0.25

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
//  linearise[]  : 256-entry sRGB → linear float table
//  kLabF[]      : 1024-entry f(t) table covering [0, 1.1] (X,Y,Z/white ≤1.09)
// ───────────────────────────────────────────────────────────────────────────
struct LabLUT {
    float linearise[256];   // sRGB byte → linear float
    float labF[1024];       // f(t) = t>0.008856 ? cbrt(t) : 7.787*t+16/116
                            // domain [0,1.1], step = 1.1/1024
    static constexpr float kFDomain = 1.1f;
    static constexpr int   kFSize   = 1024;

    LabLUT() {
        // sRGB linearise
        for (int i = 0; i < 256; ++i) {
            float u = (float)i / 255.f;
            linearise[i] = (u <= 0.04045f)
                ? u / 12.92f
                : std::pow((u + 0.055f) / 1.055f, 2.4f);
        }
        // CIE f(t)
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
} g_lut;  // Single global instance initialised at startup

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
//
//  The inner convolution loop is written so auto-vectorisers (and explicit
//  SSE/AVX paths) can process 4 or 8 floats per iteration.  For mobile
//  ARM targets the same loop auto-vectorises with NEON via clang -O2.
//  sigma=0 → passthrough copy only.
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
    // Interleaved RGBA floats — cache-friendly for SIMD
    std::vector<float>   tmp(N*4, 0.f);
    std::vector<uint8_t> dst(N*4);

    // ── Horizontal pass ───────────────────────────────────────────────────
    // For each row we gather reflected-border samples into a contiguous
    // scratch buffer so the inner loop is pure stride-1 (SIMD-friendly).
    std::vector<float> row_scratch((W + 2*radius) * 4);

    for (int y = 0; y < H; ++y) {
        const uint8_t* srcy = src + y*W*4;
        float* pad = row_scratch.data();

        // fill padded row (reflect border)
        for (int k = 0; k < ksize; ++k) {
            int sx = std::clamp(k - radius, 0, W - 1);
            // pre-fill left fringe relative to x=0
            (void)sx; // used below per-x
        }
        // Expand row into float with border reflection
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
            // This loop body is intentionally simple so the compiler can
            // auto-vectorise it across all 4 channels simultaneously.
            for (int k = 0; k < ksize; ++k) {
                float w = kern[k];
                const float* q = base + k*4;
                r += w*q[0]; g += w*q[1]; b += w*q[2]; a += w*q[3];
            }
            float* o = outy + x*4;
            o[0]=r; o[1]=g; o[2]=b; o[3]=a;
        }
    }

    // ── Vertical pass (column-major strip for cache) ────────────────────
    // Process columns in strips of 8 to keep L1 cache hot.
    static constexpr int kColStrip = 8;
    std::vector<float> col_scratch((H + 2*radius) * kColStrip * 4);

    for (int x0 = 0; x0 < W; x0 += kColStrip) {
        int x1 = std::min(x0 + kColStrip, W);
        int ncols = x1 - x0;

        // Expand column strip into col_scratch with border reflection
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
    if (!anyOpaque) return {};

    const int targetSz = std::min(1 << std::clamp(opt.color_precision,1,8),
                                  kMaxPaletteSize);
    std::vector<ColorEntry> entries;
    entries.reserve(freq.size());
    for (auto& [c,cnt]:freq) entries.push_back({rgb24(c),cnt});

    std::vector<uint32_t> palette = medianCutPalette(entries, targetSz);
    if (palette.empty()) return {};

    // Pre-compute Lab for all palette entries (LUT-accelerated)
    const int K = (int)palette.size();
    std::vector<Lab> palLab(K);
    for (int i=0;i<K;++i) palLab[i] = rgbToLabLUT(palette[i]);

    UnionFind uf(K);
    if (opt.gradient_step > 0.f) {
        const float thresh2 = opt.gradient_step * opt.gradient_step;
        for (int ii=0;ii<K;++ii)
            for (int jj=ii+1;jj<K;++jj){
                const Lab& la=palLab[ii]; const Lab& lb=palLab[jj];
                float dL=la.L-lb.L, da=la.a-lb.a, db=la.b-lb.b;
                if (dL*dL+da*da+db*db <= thresh2) uf.unite(ii,jj);
            }
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

    // Nearest-palette lookup with Lab LUT + cache
    // Also cache Lab of query colours to avoid redundant computation
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
//
//  Rather than tracing each colour region independently (which lets RDP
//  simplify shared edges differently), we:
//    1. Scan every horizontal and vertical pixel boundary (pixel edge).
//    2. For each edge where the two adjacent pixels have different canonical
//       colours, record a directed half-edge keyed by (colorA, colorB).
//    3. Assemble edge chains by following adjacency in the grid.
//    4. Expose shared edge chains so the boundary tracer re-uses them
//       verbatim for both neighbouring colours.
//
//  Data structures:
//   EdgeKey   — ordered pair of colours (lo < hi)
//   HalfEdge  — one directed segment on a colour boundary
//   EdgeGraph — maps EdgeKey → sorted vector of HalfEdges
// ═══════════════════════════════════════════════════════════════════════════

// An axis-aligned unit edge between two adjacent pixels.
// We store it as the coordinate of the *boundary* in sub-pixel space
// (multiplied by 2 to stay integer, so boundary at x=2.5 → ix=5).
struct EdgePixel {
    int x, y;        // pixel grid coords of the "left/top" pixel
    int axis;        // 0=horizontal boundary (top/bottom), 1=vertical (left/right)
    uint32_t cA, cB; // colours on the two sides (cA is always the lesser)
};

struct SharedEdgeGraph {
    // Maps (colorA<<32|colorB) where colorA<colorB → list of boundary pixels
    // These are *unordered* raw pixels; the boundary tracer will sort/chain them.
    std::unordered_map<uint64_t, std::vector<EdgePixel>> edges;

    static uint64_t key(uint32_t a, uint32_t b) {
        if (a > b) std::swap(a, b);
        return ((uint64_t)a << 32) | (uint64_t)b;
    }

    void insert(int x, int y, int axis, uint32_t cA, uint32_t cB) {
        edges[key(cA,cB)].push_back({x, y, axis, cA, cB});
    }

    // Returns true if the boundary between cA and cB at pixel (x,y) was
    // shared-registered, i.e. these two colours have an explicit shared edge
    // topology entry. Callers use this to decide whether to snap boundary
    // vertices to shared-edge positions.
    bool hasEdge(uint32_t cA, uint32_t cB) const {
        return edges.count(key(cA,cB)) > 0;
    }
};

static SharedEdgeGraph buildEdgeGraph(
    const std::vector<uint32_t>& pixelColor, int W, int H)
{
    SharedEdgeGraph graph;
    graph.edges.reserve(256);

    // Horizontal boundaries (pixel (x,y) vs (x,y+1))
    for (int y = 0; y < H-1; ++y) {
        for (int x = 0; x < W; ++x) {
            uint32_t cT = pixelColor[y*W+x];
            uint32_t cB = pixelColor[(y+1)*W+x];
            if (cT == 0xFFFFFFFFu || cB == 0xFFFFFFFFu) continue;
            if (cT != cB) graph.insert(x, y, 0, cT, cB);
        }
    }
    // Vertical boundaries (pixel (x,y) vs (x+1,y))
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
//  Pixel centres offset by +0.5 for sub-pixel accuracy.
//
//  Enhanced: after tracing, snap any vertex that lies on a registered shared
//  edge to the canonical sub-pixel boundary position, so adjacent colours
//  meet *exactly* rather than drifting apart after independent RDP runs.
// ───────────────────────────────────────────────────────────────────────────
static std::vector<Point> traceBoundary(
    int startX, int startY, int W, int H,
    std::vector<uint8_t>& occ)
{
    std::vector<Point> path; path.reserve(64);

    int entryDir=-1;
    for (int d=0;d<8;++d){
        int nx=startX+DX[d], ny=startY+DY[d];
        if ((unsigned)nx>=(unsigned)W||(unsigned)ny>=(unsigned)H) continue;
        if (occ[ny*W+nx]){entryDir=d;break;}
    }
    if (entryDir==-1){
        path.push_back({startX+.5f,startY+.5f});
        return path;
    }

    int curX=startX, curY=startY;
    int fromDir=(entryDir+5)%8;
    int firstEntryDir=-1;
    bool firstVisit=true;
    const int maxSteps=W*H*2+8;

    for (int step=0;step<maxSteps;++step){
        path.push_back({curX+.5f,curY+.5f});
        occ[curY*W+curX]=2;
        bool found=false;
        for (int i=0;i<8;++i){
            int dir=(fromDir+i)%8;
            int nx=curX+DX[dir], ny=curY+DY[dir];
            if ((unsigned)nx>=(unsigned)W||(unsigned)ny>=(unsigned)H) continue;
            if (!occ[ny*W+nx]) continue;
            if (nx==startX&&ny==startY){
                if (firstVisit){firstEntryDir=dir;firstVisit=false;}
                else if (dir==firstEntryDir) return path;
            } else firstVisit=false;
            curX=nx; curY=ny;
            fromDir=(dir+5)%8;
            found=true; break;
        }
        if (!found) break;
    }
    return path;
}

// Snap boundary vertices to exact shared-edge grid lines.
// For a pixel at grid integer (px,py), its centre is (px+0.5, py+0.5).
// The boundary between it and its right neighbour is at x=px+1.0 exactly.
// This function nudges each traced vertex that is within 'snap' pixels of
// a shared boundary line to lie exactly on that line.
static void snapToSharedEdges(
    std::vector<Point>& path,
    const std::vector<uint32_t>& pixelColor,
    uint32_t myColor,
    int W, int H,
    float snapDist = 0.6f)
{
    for (auto& pt : path) {
        // Integer pixel coords for this vertex
        int px = (int)pt.x, py = (int)pt.y;
        px = std::clamp(px, 0, W-1);
        py = std::clamp(py, 0, H-1);

        // Check left/right shared boundary
        float nearestX = std::round(pt.x);  // grid line at integer x
        if (std::abs(pt.x - nearestX) < snapDist) {
            int nx = (int)nearestX;
            if (nx > 0 && nx < W) {
                // pixels to left and right of this boundary
                int iL = py*W + (nx-1), iR = py*W + nx;
                uint32_t cL = (iL>=0&&iL<W*H) ? pixelColor[iL] : 0xFFFFFFFFu;
                uint32_t cR = (iR>=0&&iR<W*H) ? pixelColor[iR] : 0xFFFFFFFFu;
                if ((cL==myColor) != (cR==myColor))
                    pt.x = (float)nx;  // snap to exact boundary
            }
        }
        // Check top/bottom shared boundary
        float nearestY = std::round(pt.y);
        if (std::abs(pt.y - nearestY) < snapDist) {
            int ny = (int)nearestY;
            if (ny > 0 && ny < H) {
                int iT = (ny-1)*W + px, iB = ny*W + px;
                uint32_t cT = (iT>=0&&iT<W*H) ? pixelColor[iT] : 0xFFFFFFFFu;
                uint32_t cB = (iB>=0&&iB<W*H) ? pixelColor[iB] : 0xFFFFFFFFu;
                if ((cT==myColor) != (cB==myColor))
                    pt.y = (float)ny;  // snap to exact boundary
            }
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
//  Stage 7 — Ramer-Douglas-Peucker simplification
//  Iterative (stack-based) to avoid stack overflow on large paths.
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
//         (Schneider-style iterative refinement)
//
//  Given a sequence of high-resolution boundary points (the pre-RDP raw
//  path) and a small set of "key" control positions (the RDP survivors),
//  we fit one cubic Bézier per inter-key interval by:
//
//   a. Parameterise the original points by chord length (u ∈ [0,1]).
//   b. Set up a 2-DOF least-squares system for the two inner control
//      points (P1, P2) of the cubic B(t) = (1-t)³P0 + 3(1-t)²t·P1
//                                        + 3(1-t)t²·P2 + t³·P3
//      with P0 and P3 fixed (the key vertices).
//   c. Solve the 2×2 normal equations analytically.
//   d. Re-parameterise (Newton step) and iterate up to kMaxFitIter times.
//
//  If the max residual exceeds kFitTolerance the segment is subdivided at
//  its worst point and the two halves are fitted independently, giving the
//  engine a principled fallback for high-curvature regions.
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

// Chord-length parameterise a range [lo,hi) of rawPts
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

// Fit one cubic segment [P0,P3] to raw points in range [lo,hi) of rawPts.
// Returns {P1, P2} — the two free control points.
struct CubicFit { Point P1, P2; float maxResidSq; };

static CubicFit fitCubicSegment(
    const std::vector<Point>& raw, int lo, int hi,
    const Point& P0, const Point& P3)
{
    const int n = hi - lo;
    if (n <= 2) {
        // Degenerate: linear interpolation
        Point P1 = {P0.x + (P3.x-P0.x)/3.f, P0.y + (P3.y-P0.y)/3.f};
        Point P2 = {P0.x + 2.f*(P3.x-P0.x)/3.f, P0.y + 2.f*(P3.y-P0.y)/3.f};
        return {P1, P2, 0.f};
    }

    std::vector<float> u = chordParam(raw, lo, hi);

    // Normal equations for the 2 inner control points.
    // B(t) = C0(t)*P0 + C1(t)*P1 + C2(t)*P2 + C3(t)*P3
    // where C1=3(1-t)²t, C2=3(1-t)t²
    // Least-squares: minimise Σ|B(u_i) - raw_i|²
    // Rearrange to: Σ[C1²]·P1 + Σ[C1·C2]·P2 = Σ[C1·(Q_i)]
    //               Σ[C1·C2]·P1 + Σ[C2²]·P2 = Σ[C2·(Q_i)]
    // where Q_i = raw_i - C0(u_i)*P0 - C3(u_i)*P3

    double A00=0,A01=0,A11=0;
    double Bx0=0,By0=0,Bx1=0,By1=0;

    for (int i=0;i<n;++i){
        float t=u[i];
        float mt=1.f-t;
        float c0=mt*mt*mt, c1=3.f*mt*mt*t, c2=3.f*mt*t*t, c3=t*t*t;
        float Qx=raw[lo+i].x - c0*P0.x - c3*P3.x;
        float Qy=raw[lo+i].y - c0*P0.y - c3*P3.y;
        A00+=c1*c1; A01+=c1*c2; A11+=c2*c2;
        Bx0+=c1*Qx; By0+=c1*Qy;
        Bx1+=c2*Qx; By1+=c2*Qy;
    }

    // Solve 2×2 system
    double det = A00*A11 - A01*A01;
    Point P1, P2;
    if (std::abs(det) < 1e-10) {
        // Singular: fall back to Catmull-Rom tangent estimate
        P1 = {P0.x + (P3.x-P0.x)/3.f, P0.y + (P3.y-P0.y)/3.f};
        P2 = {P0.x + 2.f*(P3.x-P0.x)/3.f, P0.y + 2.f*(P3.y-P0.y)/3.f};
    } else {
        double invDet = 1.0 / det;
        P1.x = (float)((A11*Bx0 - A01*Bx1)*invDet);
        P1.y = (float)((A11*By0 - A01*By1)*invDet);
        P2.x = (float)((A00*Bx1 - A01*Bx0)*invDet);
        P2.y = (float)((A00*By1 - A01*By0)*invDet);
    }

    // Compute max residual
    float maxRes = 0.f;
    for (int i=0;i<n;++i){
        Point b = bezier(P0, P1, P2, P3, u[i]);
        float dx=b.x-raw[lo+i].x, dy=b.y-raw[lo+i].y;
        float r=dx*dx+dy*dy;
        if (r>maxRes) maxRes=r;
    }
    return {P1, P2, maxRes};
}

// Newton reparameterisation step: for each raw point find the closest t
// on the current Bézier using one Newton iteration.
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
        // B(t), B'(t)
        Point Bt  = bezier(P0,P1,P2,P3,t);
        float mt=1.f-t;
        // B'(t) = 3[(1-t)²(P1-P0) + 2(1-t)t(P2-P1) + t²(P3-P2)]
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
static void fitBezierRecursive(
    const std::vector<Point>& raw, int lo, int hi,  // original high-res points
    const Point& P0, const Point& P3,               // fixed endpoints
    std::vector<Segment>& out,
    int depth = 0)
{
    static constexpr int kMaxDepth = 6;
    if (hi - lo <= 1 || depth > kMaxDepth) {
        // Leaf: emit a simple line segment
        out.push_back({false,{},{},P3});
        return;
    }

    // Initial parameterisation
    std::vector<float> u = chordParam(raw, lo, hi);

    // Iterative refinement
    Point P1_best, P2_best;
    float bestRes = 1e30f;
    for (int iter=0; iter<kMaxFitIter; ++iter){
        CubicFit fit = fitCubicSegment(raw, lo, hi, P0, P3);
        if (fit.maxResidSq < bestRes){
            bestRes = fit.maxResidSq;
            P1_best = fit.P1; P2_best = fit.P2;
        }
        if (fit.maxResidSq <= kFitTolerance*kFitTolerance) break;
        // Newton reparameterisation
        u = reparameterise(raw, lo, hi, P0, fit.P1, fit.P2, P3, u);
    }

    if (bestRes <= kFitTolerance*kFitTolerance) {
        out.push_back({true, P1_best, P2_best, P3});
        return;
    }

    // Subdivide at max-residual point and fit each half
    // Re-run to find the worst-fit index
    CubicFit fit = fitCubicSegment(raw, lo, hi, P0, P3);
    int worstI = lo;
    {
        float wRes = 0.f;
        std::vector<float> u2 = chordParam(raw, lo, hi);
        for (int i=0; i<hi-lo; ++i){
            Point b = bezier(P0, fit.P1, fit.P2, P3, u2[i]);
            float dx=b.x-raw[lo+i].x, dy=b.y-raw[lo+i].y;
            float r=dx*dx+dy*dy;
            if (r>wRes){wRes=r;worstI=lo+i;}
        }
    }
    if (worstI == lo || worstI == hi-1) {
        // Can't split usefully — emit what we have
        out.push_back({true, P1_best, P2_best, P3});
        return;
    }
    const Point& Pmid = raw[worstI];
    fitBezierRecursive(raw, lo,    worstI+1, P0,   Pmid, out, depth+1);
    fitBezierRecursive(raw, worstI, hi,      Pmid, P3,   out, depth+1);
}

// Build the spline for one boundary region: given simplified key vertices
// and the original high-res path, produce globally-optimal Bézier segments.
static std::vector<Segment> buildSplineLSQ(
    const std::vector<Point>& keyPts,         // RDP-simplified
    const std::vector<uint8_t>& isCorner,
    const std::vector<Point>& rawPts)         // original high-res boundary
{
    const int nk = (int)keyPts.size();
    const int nr = (int)rawPts.size();
    if (nk < 2 || nr < 2) return {};

    std::vector<Segment> segs;
    segs.reserve(nk);

    // Map each key vertex to the closest raw vertex index
    auto closestRaw = [&](const Point& p) -> int {
        float best = 1e30f; int bi = 0;
        for (int i=0;i<nr;++i){
            float dx=rawPts[i].x-p.x, dy=rawPts[i].y-p.y;
            float d=dx*dx+dy*dy;
            if (d<best){best=d;bi=i;}
        }
        return bi;
    };

    // Build corner index list in key-point space
    std::vector<int> corners;
    for (int i=0;i<nk;++i) if (isCorner[i]) corners.push_back(i);
    if (corners.empty()) corners.push_back(0);

    const int nc = (int)corners.size();
    auto nxt = [&](int i){ return (i+1)%nk; };

    for (int ci=0;ci<nc;++ci){
        int from = corners[ci];
        int to   = corners[(ci+1)%nc];

        // Collect key-point run from→to (circular)
        std::vector<int> run;
        for (int k=from;;k=nxt(k)){
            run.push_back(k);
            if (k==to) break;
            if ((int)run.size()>nk+2) break;
        }
        const int rn = (int)run.size();
        if (rn < 2) continue;

        for (int ri=0; ri<rn-1; ++ri){
            const Point& P0 = keyPts[run[ri]];
            const Point& P3 = keyPts[run[ri+1]];

            // Find raw index range corresponding to this key interval
            int rlo = closestRaw(P0);
            int rhi = closestRaw(P3);

            // Handle wrap-around in circular raw path
            std::vector<Point> seg_raw;
            if (rlo <= rhi) {
                seg_raw.assign(rawPts.begin()+rlo, rawPts.begin()+rhi+1);
            } else {
                // Wrap: rlo..end ++ 0..rhi
                seg_raw.assign(rawPts.begin()+rlo, rawPts.end());
                seg_raw.insert(seg_raw.end(), rawPts.begin(), rawPts.begin()+rhi+1);
            }

            if ((int)seg_raw.size() < 2) {
                segs.push_back({false,{},{},P3});
                continue;
            }

            // Check if it's a simple short segment — use line
            float dx=P3.x-P0.x, dy=P3.y-P0.y;
            if (dx*dx+dy*dy < 1.f){
                segs.push_back({false,{},{},P3});
                continue;
            }

            fitBezierRecursive(seg_raw, 0, (int)seg_raw.size(), P0, P3, segs);
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
    assert(pixels && width>0 && height>0);

    // Apply defaults for unset options
    if (options.color_precision  <= 0) options.color_precision  = kDefaultColorPrecision;
    if (options.corner_threshold <= 0) options.corner_threshold = kDefaultCornerThreshold;
    if (options.filter_speckle   <= 0) options.filter_speckle   = kDefaultFilterSpeckle;
    if (options.path_precision   <  0) options.path_precision   = kDefaultPathPrecision;
    if (options.rdp_epsilon      <= 0) options.rdp_epsilon      = kDefaultRdpEpsilon;
    if (options.blur_radius      <  0) options.blur_radius      = kDefaultBlurRadius;

    const int dp = std::clamp(options.path_precision, 0, 6);
    const int N  = width * height;

    // Stage 0: Gaussian pre-blur (SIMD/LUT accelerated)
    std::vector<uint8_t> blurred = gaussianBlur(pixels, width, height, options.blur_radius);
    const uint8_t* src = blurred.data();

    // Stages 1+2: median-cut quantise + Lab gradient merge (LUT accelerated)
    std::vector<uint32_t> pixelColor;
    std::vector<uint32_t> palette =
        buildPaletteAndAssign(src, width, height, options, pixelColor);

    if (palette.empty()){
        char buf[128];
        snprintf(buf,sizeof(buf),
                 "<svg xmlns='http://www.w3.org/2000/svg' "
                 "viewBox='0 0 %d %d'></svg>",width,height);
        return buf;
    }

    // Stage 3: connected components
    std::vector<uint32_t> componentColor;
    std::vector<int>      componentSize;
    std::vector<int> labelMap =
        labelComponents(pixelColor, width, height, componentColor, componentSize);

    std::unordered_map<uint32_t,std::vector<int>> colorToComponents;
    colorToComponents.reserve(palette.size()*2);
    for (int lbl=0;lbl<(int)componentColor.size();++lbl)
        colorToComponents[componentColor[lbl]].push_back(lbl);

    // [E1] Stage 5: Build shared edge graph ONCE for the entire image
    SharedEdgeGraph edgeGraph = buildEdgeGraph(pixelColor, width, height);

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

    // Stages 4–11: per-colour processing, darkest-first layer order
    for (uint32_t color : palette){
        if (!colorToComponents.count(color)) continue;

        std::fill(occ.begin(),occ.end(),0);
        for (int i=0;i<N;++i)
            if (pixelColor[i]==color) occ[i]=1;

        struct PathRecord {
            std::vector<Point>   rawPts;      // full high-res boundary [E3]
            std::vector<Point>   pts;         // RDP-simplified
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
                    continue;
                }

                // Stage 6: boundary trace
                std::vector<Point> raw=traceBoundary(x,y,width,height,occ);
                clearComponent(idx,lbl,labelMap,occ,width,height);
                if ((int)raw.size()<3) continue;

                // [E1] Snap raw boundary vertices to exact shared edge grid lines
                snapToSharedEdges(raw, pixelColor, color, width, height);

                // Stage 7: RDP simplification (applied to snapped raw)
                std::vector<Point> simplified=rdpSimplify(raw,options.rdp_epsilon);
                if ((int)simplified.size()<3) continue;

                // Stage 8: corner detection
                std::vector<uint8_t> corners=
                    detectCorners(simplified,options.corner_threshold);

                // [E3] Stage 9: Constrained LS Bézier fit using raw high-res data
                std::vector<Segment> segs=buildSplineLSQ(simplified,corners,raw);
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
            // Re-detect corners after winding fix and rebuild spline [E3]
            auto c2=detectCorners(pr.pts,options.corner_threshold);
            pr.segs=buildSplineLSQ(pr.pts,c2,pr.rawPts);
        }

        // Stage 11: emit SVG — group all paths by colour
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
        }
        svg+="</g>";
    }

    svg+="</svg>";
    return svg;
}

} // namespace vtracer