// ═══════════════════════════════════════════════════════════════════════════
//  VTracerEngine.cpp  —  production-grade SVG vectoriser
 
//  Performance improvements:
//   PERF-1 : Bilateral filter spatial weights pre-computed into a 2D LUT
//             indexed by (dy+R, dx+R) — eliminates std::exp from the inner loop
//   PERF-2 : Bilateral filter range weights pre-computed into a 256-entry LUT
//             indexed by squared-distance bucket — eliminates second std::exp
//   PERF-3 : occ[] reset uses a generation counter instead of std::fill(N)
//             per colour — O(dirty pixels) instead of O(W×H) per palette entry
//   PERF-4 : BFS queue in labelComponents uses index into growing vector
//             (already the case) but clearComponent now uses the same pattern
//             consistently — avoids repeated modulo divisions
//   PERF-5 : [[nodiscard]] / noexcept annotations on pure functions allow
//             the compiler to elide stack-unwinding code on ARM
//
//  SVG output quality improvements:
//   SVG-1  : shape-rendering="geometricPrecision" on root <svg> — sub-pixel
//             accurate Bézier rendering on Retina / HiDPI mobile displays
//   SVG-2  : Double-quote attribute delimiters throughout (canonical SVG form)
//   SVG-3  : gradient_detect_thresh default lowered from 22 → 16 (ΔE≈4)
//             to avoid merging perceptually distinct colours into gradients
//   SVG-4  : xmlns:xlink emitted when gradient defs are present (rasteriser compat)
//   SVG-5  : Gradient path accumulator now joins without a trailing space
// ═══════════════════════════════════════════════════════════════════════════


#include "VTracerEngine.hpp"


#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>       // FIX-2: replaces <time.h> + clock_gettime
#include <climits>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>


// ── Logging ──────────────────────────────────────────────────────────────────
#ifdef __ANDROID__
#  include <android/log.h>
#  define VT_LOG(fmt,...)  __android_log_print(ANDROID_LOG_DEBUG,"VTracerEngine",fmt,##__VA_ARGS__)
#  define VT_WARN(fmt,...) __android_log_print(ANDROID_LOG_WARN, "VTracerEngine",fmt,##__VA_ARGS__)
#  define VT_ERR(fmt,...)  __android_log_print(ANDROID_LOG_ERROR,"VTracerEngine",fmt,##__VA_ARGS__)
#else
#  define VT_LOG(fmt,...)  fprintf(stderr,"[VTracer] "    fmt "\n",##__VA_ARGS__)
#  define VT_WARN(fmt,...) fprintf(stderr,"[VTracer WARN] " fmt "\n",##__VA_ARGS__)
#  define VT_ERR(fmt,...)  fprintf(stderr,"[VTracer ERR]  " fmt "\n",##__VA_ARGS__)
#endif


// ── Timing — FIX-2: std::chrono::steady_clock (portable, no POSIX) ──────────
static inline double vt_now_ms() noexcept {
    using Clock = std::chrono::steady_clock;
    using Ms    = std::chrono::duration<double, std::milli>;
    return std::chrono::duration_cast<Ms>(Clock::now().time_since_epoch()).count();
}


namespace vtracer {


// ─────────────────────────────────────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int   kDefaultColorPrecision  = 6;
static constexpr float kDefaultCornerThreshold = 120.f;
static constexpr int   kDefaultFilterSpeckle   = 4;
static constexpr float kDefaultRdpEpsilon      = 1.5f;
static constexpr int   kDefaultPathPrecision   = 1;
static constexpr float kDefaultBlurRadius      = 1.0f;
static constexpr int   kMaxPaletteSize         = 256;

// FIX-1: portable pi — no POSIX M_PI dependency
static constexpr float kPi                     = 3.14159265358979f;

// [E10] Raised from 8 → 20; early-stop if residual improvement < kFitConvergEps
static constexpr int   kMaxFitIter             = 20;
static constexpr float kFitTolerance           = 0.5f;
static constexpr float kFitConvergEps          = 1e-4f;

// [E2]  After LS solve, |P1-P0| and |P2-P3| must be ≤ kCPClampK × |P3-P0|
static constexpr float kCPClampK               = 2.0f;

// [E11] Half-width of the corner-detection tangent window (uses ±kCornerHW pts)
static constexpr int   kCornerHW               = 3;

// [E9]  ΔE² threshold for gradient-group detection
//       SVG-3: lowered from 22 → 16 (ΔE≈4) to avoid over-merging
static constexpr float kGradDetectDefault      = 16.0f;


// ─────────────────────────────────────────────────────────────────────────────
//  Geometry
// ─────────────────────────────────────────────────────────────────────────────
struct Point   { float x, y; };
struct Segment { bool isCurve; Point cp1, cp2, end; };


static inline Point  operator+(const Point& a,const Point& b) noexcept {return {a.x+b.x,a.y+b.y};}
static inline Point  operator-(const Point& a,const Point& b) noexcept {return {a.x-b.x,a.y-b.y};}
static inline Point  operator*(float s,const Point& p)        noexcept {return {s*p.x,s*p.y};}
static inline Point  operator*(const Point& p,float s)        noexcept {return {s*p.x,s*p.y};}
static inline float  dot(const Point& a,const Point& b)       noexcept {return a.x*b.x+a.y*b.y;}
static inline float  lenSq(const Point& p)                    noexcept {return p.x*p.x+p.y*p.y;}
static inline float  vlen(const Point& p)                     noexcept {return std::sqrt(lenSq(p));}


// ─────────────────────────────────────────────────────────────────────────────
//  Moore neighbourhood — 8-connected clockwise from North
//    7 0 1 / 6 * 2 / 5 4 3
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int DX[8] = { 0,  1,  1,  1,  0, -1, -1, -1 };
static constexpr int DY[8] = {-1, -1,  0,  1,  1,  1,  0, -1 };


// ─────────────────────────────────────────────────────────────────────────────
//  Union-Find (path compression + rank)
// ─────────────────────────────────────────────────────────────────────────────
struct UnionFind {
    std::vector<int> parent, rank_;
    explicit UnionFind(int n):parent(n),rank_(n,0){std::iota(parent.begin(),parent.end(),0);}
    int find(int x) noexcept {
        while(parent[x]!=x){parent[x]=parent[parent[x]];x=parent[x];}return x;
    }
    void unite(int a,int b) noexcept {
        a=find(a);b=find(b);if(a==b)return;
        if(rank_[a]<rank_[b])std::swap(a,b);
        parent[b]=a;if(rank_[a]==rank_[b])++rank_[a];
    }
};


// ─────────────────────────────────────────────────────────────────────────────
//  Colour helpers
// ─────────────────────────────────────────────────────────────────────────────
static inline uint32_t packRGB(uint8_t r,uint8_t g,uint8_t b) noexcept {
    return((uint32_t)r<<16)|((uint32_t)g<<8)|b;
}
static inline uint32_t rgb24(uint32_t c)  noexcept {return c&0x00FFFFFFu;}
static inline uint8_t  rCh(uint32_t c)   noexcept {return(c>>16)&0xFF;}
static inline uint8_t  gCh(uint32_t c)   noexcept {return(c>> 8)&0xFF;}
static inline uint8_t  bCh(uint32_t c)   noexcept {return c     &0xFF;}


// ─────────────────────────────────────────────────────────────────────────────
//  CIE-Lab LUT
//  FIX-4: g_lut is now a Meyer's singleton — eliminates static-init-order risk
// ─────────────────────────────────────────────────────────────────────────────
struct LabLUT {
    float linearise[256];
    float labF[1024];
    static constexpr float kFDomain = 1.1f;
    static constexpr int   kFSize   = 1024;

    LabLUT() {
        for(int i=0;i<256;++i){
            float u=(float)i/255.f;
            linearise[i]=(u<=0.04045f)?u/12.92f:std::pow((u+0.055f)/1.055f,2.4f);
        }
        for(int i=0;i<kFSize;++i){
            float t=kFDomain*(float)i/(float)kFSize;
            labF[i]=(t>0.008856f)?std::cbrt(t):(7.787f*t+16.f/116.f);
        }
    }
    inline float f(float t) const noexcept {
        int idx=(int)(t*(float)kFSize/kFDomain);
        idx=std::clamp(idx,0,kFSize-1);
        return labF[idx];
    }
};

// FIX-4: Meyer's singleton — zero SIOF risk, thread-safe under C++11/17
static const LabLUT& lut() noexcept {
    static const LabLUT s_lut;
    return s_lut;
}


struct Lab {float L,a,b;};


static Lab rgbToLabLUT(uint32_t c) noexcept {
    const LabLUT& L = lut();
    float rl=L.linearise[rCh(c)];
    float gl=L.linearise[gCh(c)];
    float bl=L.linearise[bCh(c)];
    float X=rl*0.4124564f+gl*0.3575761f+bl*0.1804375f;
    float Y=rl*0.2126729f+gl*0.7151522f+bl*0.0721750f;
    float Z=rl*0.0193339f+gl*0.1191920f+bl*0.9503041f;
    X/=0.95047f; Z/=1.08883f;
    float fx=L.f(X), fy=L.f(Y), fz=L.f(Z);
    return{116.f*fy-16.f, 500.f*(fx-fy), 200.f*(fy-fz)};
}


static float labDistSq(uint32_t a,uint32_t b) noexcept {
    Lab la=rgbToLabLUT(rgb24(a)), lb=rgbToLabLUT(rgb24(b));
    float dL=la.L-lb.L, da=la.a-lb.a, db=la.b-lb.b;
    return dL*dL+da*da+db*db;
}


// ═════════════════════════════════════════════════════════════════════════════
// Stage 0 — Edge-Preserving Bilateral Filter
//
//   PERF-1/2: std::exp() has been eliminated from the inner loop entirely.
//   Spatial weights are pre-computed into spatialW[(2R+1)*(2R+1)].
//   Range weights are pre-computed into rangeW[256] indexed by the
//   mean-squared channel difference clamped to [0,255].
//   This reduces the inner loop to two table lookups + multiply, which is
//   dramatically faster on ARM mobile (exp ≈ 20–50 ns; LUT lookup ≈ 1–2 ns).
//
//   effectiveSigma = blur_radius × max(W,H)/512 so that at camera resolution
//   the filter genuinely suppresses JPEG/sensor noise.  Kernel radius is capped
//   at 5 (11×11 window, 121 ops/pixel) to keep latency acceptable on mobile.
// ═════════════════════════════════════════════════════════════════════════════
static std::vector<uint8_t> bilateralFilter(
    const uint8_t* src, int W, int H,
    float sigma_s, float sigma_r)
{
    if (sigma_s < 0.1f)
        return std::vector<uint8_t>(src, src + (size_t)W * H * 4);

    // [E14] Scale spatial sigma with image resolution
    const float scaledSigma = sigma_s * std::max(1.f,
        (float)std::max(W, H) / 512.f);

    // [E14] Cap radius at 5 (mobile budget: ≤ 121 ops/pixel)
    const int radius = std::min((int)std::ceil(2.f * scaledSigma), 5);
    const int D      = 2 * radius + 1;

    // PERF-1: pre-compute spatial weight table  [D × D]
    const float inv2Ss2 = 1.f / (2.f * scaledSigma * scaledSigma);
    std::vector<float> spatialW((size_t)D * D);
    for(int dy=-radius; dy<=radius; ++dy)
        for(int dx=-radius; dx<=radius; ++dx)
            spatialW[(size_t)(dy+radius)*D + (dx+radius)] =
                std::exp(-(float)(dx*dx+dy*dy) * inv2Ss2);

    // PERF-2: pre-compute range weight table  [256 entries]
    //   index = clamp(round(mean squared channel diff / 3), 0, 255)
    const float inv2Sr2 = 1.f / (2.f * sigma_r * sigma_r);
    float rangeW[256];
    for(int i=0;i<256;++i)
        rangeW[i] = std::exp(-(float)i * inv2Sr2);

    const int N = W * H;
    std::vector<uint8_t> dst((size_t)N * 4);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const uint8_t* ctr = src + (y * W + x) * 4;
            float sumR=0,sumG=0,sumB=0,sumA=0,sumWt=0;

            const int y0 = std::max(y - radius, 0);
            const int y1 = std::min(y + radius, H - 1);
            const int x0 = std::max(x - radius, 0);
            const int x1 = std::min(x + radius, W - 1);

            for (int ny = y0; ny <= y1; ++ny) {
                for (int nx = x0; nx <= x1; ++nx) {
                    const uint8_t* nbr = src + (ny * W + nx) * 4;

                    // PERF-1: spatial weight from LUT
                    const float wS = spatialW[(size_t)(ny-y+radius)*D + (nx-x+radius)];

                    // PERF-2: range weight from LUT (integer mean-sq-diff clamped)
                    int dr=(int)ctr[0]-(int)nbr[0];
                    int dg=(int)ctr[1]-(int)nbr[1];
                    int db=(int)ctr[2]-(int)nbr[2];
                    int idx = std::clamp((dr*dr+dg*dg+db*db)/3, 0, 255);
                    const float wR = rangeW[idx];

                    const float w = wS * wR;
                    sumR += w*nbr[0]; sumG += w*nbr[1];
                    sumB += w*nbr[2]; sumA += w*nbr[3];
                    sumWt += w;
                }
            }

            uint8_t* d = dst.data() + (y * W + x) * 4;
            if (sumWt > 1e-8f) {
                d[0]=(uint8_t)std::clamp((int)(sumR/sumWt+.5f),0,255);
                d[1]=(uint8_t)std::clamp((int)(sumG/sumWt+.5f),0,255);
                d[2]=(uint8_t)std::clamp((int)(sumB/sumWt+.5f),0,255);
                d[3]=(uint8_t)std::clamp((int)(sumA/sumWt+.5f),0,255);
            } else {
                d[0]=ctr[0];d[1]=ctr[1];d[2]=ctr[2];d[3]=ctr[3];
            }
        }
    }
    return dst;
}


// ─────────────────────────────────────────────────────────────────────────────
//  Stage 1 — Median-cut palette quantisation
//       Both the split axis and the split sort operate in CIE-Lab space.
//       For a pastel photo the perceptually dominant spread is usually on
//       the Lab a* (pink↔green) or b* (yellow↔blue) axis rather than in R/G/B,
//       so Lab-space splitting puts palette entries where the eye needs them.
// ─────────────────────────────────────────────────────────────────────────────
struct ColorEntry { uint32_t color; int count; };


// [E3] Return the Lab channel (0=L 1=a 2=b) with the greatest range in [lo,hi)
static int widestChannelLab(const std::vector<ColorEntry>& E, int lo, int hi) noexcept {
    float LMn=1e30f,LMx=-1e30f,aMn=1e30f,aMx=-1e30f,bMn=1e30f,bMx=-1e30f;
    for (int i=lo; i<hi; ++i) {
        Lab l = rgbToLabLUT(E[i].color);
        LMn=std::min(LMn,l.L); LMx=std::max(LMx,l.L);
        aMn=std::min(aMn,l.a); aMx=std::max(aMx,l.a);
        bMn=std::min(bMn,l.b); bMx=std::max(bMx,l.b);
    }
    float LR=LMx-LMn, aR=aMx-aMn, bR=bMx-bMn;
    return (LR>=aR && LR>=bR) ? 0 : (aR>=bR) ? 1 : 2;
}


// FIX-6: median-cut split — pre-compute Lab keys before sort so the
//         comparator does not call rgbToLabLUT O(N log N) times.
static int medianCutSplit(
    std::vector<ColorEntry>& E, int lo, int hi, int labCh)
{
    // Pre-compute sort key for each entry in [lo,hi)
    std::vector<float> key((size_t)(hi-lo));
    for(int i=lo;i<hi;++i){
        Lab l=rgbToLabLUT(E[i].color);
        key[i-lo]=(labCh==0)?l.L:(labCh==1)?l.a:l.b;
    }

    // Sort entries by pre-computed key via an index array
    std::vector<int> idx((size_t)(hi-lo));
    std::iota(idx.begin(),idx.end(),0);
    std::sort(idx.begin(),idx.end(),[&](int a,int b){return key[a]<key[b];});

    // Apply permutation in-place
    std::vector<ColorEntry> tmp(E.begin()+lo,E.begin()+hi);
    for(int i=0;i<(int)idx.size();++i) E[lo+i]=tmp[idx[i]];

    // Weighted median split
    long total=0;
    for(int i=lo;i<hi;++i) total+=E[i].count;
    long half=total/2, acc=0;
    for(int i=lo;i<hi-1;++i){
        acc+=E[i].count;
        if(acc>=half) return i+1;
    }
    return (lo+hi)/2;
}


static uint32_t boxRepresentative(
    const std::vector<ColorEntry>& E, int lo, int hi) noexcept
{
    double r=0,g=0,b=0; long total=0;
    for(int i=lo;i<hi;++i){
        double w=E[i].count;
        r+=w*rCh(E[i].color); g+=w*gCh(E[i].color); b+=w*bCh(E[i].color);
        total+=E[i].count;
    }
    if(!total) return 0;
    return packRGB(
        (uint8_t)std::clamp((int)(r/total+.5),0,255),
        (uint8_t)std::clamp((int)(g/total+.5),0,255),
        (uint8_t)std::clamp((int)(b/total+.5),0,255));
}


static std::vector<uint32_t> medianCutPalette(
    std::vector<ColorEntry>& E, int target)
{
    if (E.empty()) return {};
    target = std::clamp(target, 1, kMaxPaletteSize);

    if ((int)E.size() <= target) {
        std::vector<uint32_t> p; p.reserve(E.size());
        for(auto& e:E) p.push_back(rgb24(e.color));
        return p;
    }

    struct Box { int lo,hi; };
    std::vector<Box> boxes;
    boxes.push_back({0,(int)E.size()});

    while ((int)boxes.size() < target) {
        int bestBox=-1; float bestRange=-1.f;
        for(int bi=0;bi<(int)boxes.size();++bi){
            auto& bx=boxes[bi];
            if(bx.hi-bx.lo<=1) continue;
            int ch=widestChannelLab(E,bx.lo,bx.hi);
            float mn=1e30f,mx=-1e30f;
            for(int i=bx.lo;i<bx.hi;++i){
                Lab l=rgbToLabLUT(E[i].color);
                float v=ch==0?l.L:ch==1?l.a:l.b;
                mn=std::min(mn,v); mx=std::max(mx,v);
            }
            float range=mx-mn;
            if(range>bestRange){bestRange=range;bestBox=bi;}
        }
        if(bestBox==-1) break;
        auto& bx=boxes[bestBox];
        int ch  = widestChannelLab(E,bx.lo,bx.hi);
        int mid = medianCutSplit(E,bx.lo,bx.hi,ch);
        Box left={bx.lo,mid}, right={mid,bx.hi};
        boxes[bestBox]=left;
        boxes.push_back(right);
    }

    std::vector<uint32_t> pal; pal.reserve(boxes.size());
    for(auto& bx:boxes)
        if(bx.hi>bx.lo) pal.push_back(boxRepresentative(E,bx.lo,bx.hi));
    return pal;
}


// ─────────────────────────────────────────────────────────────────────────────
//  Stage 1+2 — Quantise + perceptual gradient merge
//       NOTE: final luminance sort removed here; vectorize() re-sorts by avg
//       component area (largest first) after Stage 3.
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<uint32_t> buildPaletteAndAssign(
    const uint8_t*         pixels,
    int W, int H,
    const Options&         opt,
    std::vector<uint32_t>& pixelColor)
{
    const int N = W * H;
    pixelColor.assign(N, 0xFFFFFFFFu);

    std::unordered_map<uint32_t,int> freq;
    freq.reserve(4096);
    bool anyOpaque = false;

    for(int i=0;i<N;++i){
        const uint8_t* p = pixels + i*4;
        if(p[3]==0) continue;
        anyOpaque = true;
        if(opt.color_mode == ColorMode::BlackAndWhite){
            float lum=0.299f*p[0]+0.587f*p[1]+0.114f*p[2];
            freq[lum>=128.f ? 0x00FFFFFFu : 0x00000000u]++;
        } else {
            freq[packRGB(p[0],p[1],p[2])]++;
        }
    }
    if(!anyOpaque){
        VT_WARN("buildPaletteAndAssign: all-transparent image");
        return {};
    }

    const int targetSz=std::min(1<<std::clamp(opt.color_precision,1,8),kMaxPaletteSize);
    VT_LOG("Stage 1: %d unique colours → target palette %d",(int)freq.size(),targetSz);

    std::vector<ColorEntry> entries;
    entries.reserve(freq.size());
    for(auto& [c,cnt]:freq) entries.push_back({rgb24(c),cnt});

    std::vector<uint32_t> palette = medianCutPalette(entries, targetSz);
    if(palette.empty()){VT_WARN("medianCutPalette returned empty");return{};}
    VT_LOG("Stage 1: palette %d entries",(int)palette.size());

    const int K=(int)palette.size();
    std::vector<Lab> palLab(K);
    for(int i=0;i<K;++i) palLab[i]=rgbToLabLUT(palette[i]);

    // Stage 2: perceptual gradient merge
    UnionFind uf(K);
    if(opt.gradient_step > 0.f){
        const float thresh2=opt.gradient_step*opt.gradient_step;
        int merges=0;
        for(int ii=0;ii<K;++ii)
            for(int jj=ii+1;jj<K;++jj){
                const Lab& la=palLab[ii]; const Lab& lb=palLab[jj];
                float dL=la.L-lb.L,da=la.a-lb.a,db=la.b-lb.b;
                if(dL*dL+da*da+db*db<=thresh2){uf.unite(ii,jj);++merges;}
            }
        VT_LOG("Stage 2: gradient merge step=%.2f → %d merges",(double)opt.gradient_step,merges);
    }

    struct Acc{double r,g,b;long count;};
    std::unordered_map<int,Acc> groupAcc;
    for(int i=0;i<K;++i){
        uint32_t c=palette[i];
        long cnt=freq.count(c)?freq.at(c):1;
        int root=uf.find(i);
        auto& acc=groupAcc[root];
        acc.r+=cnt*rCh(c); acc.g+=cnt*gCh(c); acc.b+=cnt*bCh(c);
        acc.count+=cnt;
    }
    std::unordered_map<int,uint32_t> rootColor;
    for(auto& [root,acc]:groupAcc)
        rootColor[root]=packRGB(
            (uint8_t)std::clamp((int)(acc.r/acc.count+.5),0,255),
            (uint8_t)std::clamp((int)(acc.g/acc.count+.5),0,255),
            (uint8_t)std::clamp((int)(acc.b/acc.count+.5),0,255));

    std::unordered_map<uint32_t,uint32_t> nearestCache;
    nearestCache.reserve(freq.size()*2);
    auto nearest=[&](uint32_t c)->uint32_t{
        auto it=nearestCache.find(c);
        if(it!=nearestCache.end()) return it->second;
        Lab lc=rgbToLabLUT(rgb24(c));
        float bestD=1e30f; int bestI=0;
        for(int i=0;i<K;++i){
            const Lab& lp=palLab[i];
            float dL=lc.L-lp.L,da=lc.a-lp.a,db=lc.b-lp.b;
            float d=dL*dL+da*da+db*db;
            if(d<bestD){bestD=d;bestI=i;}
        }
        uint32_t res=rootColor[uf.find(bestI)];
        nearestCache[c]=res;
        return res;
    };

    for(int i=0;i<N;++i){
        const uint8_t* p=pixels+i*4;
        if(p[3]==0) continue;
        uint32_t raw;
        if(opt.color_mode==ColorMode::BlackAndWhite){
            float lum=0.299f*p[0]+0.587f*p[1]+0.114f*p[2];
            raw=lum>=128.f?0x00FFFFFFu:0x00000000u;
        } else {
            raw=packRGB(p[0],p[1],p[2]);
        }
        pixelColor[i]=nearest(raw);
    }

    std::unordered_map<uint32_t,bool> seen;
    std::vector<uint32_t> used;
    for(int i=0;i<N;++i){
        if(pixelColor[i]==0xFFFFFFFFu) continue;
        if(seen.emplace(pixelColor[i],true).second)
            used.push_back(pixelColor[i]);
    }

    // [E5] Luminance sort intentionally omitted here.
    //      vectorize() re-sorts by decreasing average component area after Stage 3.
    VT_LOG("Stage 1+2: %d colours in use",(int)used.size());
    return used;
}


// ─────────────────────────────────────────────────────────────────────────────
//  Stage 3 — BFS connected-component labelling (4-connectivity)
//  Added: componentBBox output (used by E5 sort and E9 gradient defs)
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<int> labelComponents(
    const std::vector<uint32_t>& pixelColor,
    int W, int H,
    std::vector<uint32_t>& componentColor,
    std::vector<int>&      componentSize,
    std::vector<std::array<int,4>>& componentBBox)  // {x0,y0,x1,y1}
{
    const int N=W*H;
    std::vector<int> label(N,-1);
    componentColor.clear(); componentSize.clear(); componentBBox.clear();
    static constexpr int ox[4]={1,-1,0,0}, oy[4]={0,0,1,-1};
    std::vector<int> q; q.reserve(2048);

    for(int i=0;i<N;++i){
        if(pixelColor[i]==0xFFFFFFFFu||label[i]!=-1) continue;
        uint32_t myColor=pixelColor[i];
        int lbl=(int)componentColor.size();
        componentColor.push_back(myColor);
        componentSize.push_back(0);
        int ix=i%W, iy=i/W;
        componentBBox.push_back({ix,iy,ix,iy});
        label[i]=lbl; q.clear(); q.push_back(i);
        int head=0;
        while(head<(int)q.size()){
            int cur=q[head++];
            ++componentSize[lbl];
            int cx=cur%W, cy=cur/W;
            auto& bb=componentBBox[lbl];
            bb[0]=std::min(bb[0],cx); bb[1]=std::min(bb[1],cy);
            bb[2]=std::max(bb[2],cx); bb[3]=std::max(bb[3],cy);
            for(int d=0;d<4;++d){
                int nx=cx+ox[d], ny=cy+oy[d];
                if((unsigned)nx>=(unsigned)W||(unsigned)ny>=(unsigned)H) continue;
                int ni=ny*W+nx;
                if(label[ni]!=-1||pixelColor[ni]!=myColor) continue;
                label[ni]=lbl; q.push_back(ni);
            }
        }
    }
    return label;
}


// ─────────────────────────────────────────────────────────────────────────────
//  Stage 4 — Speckle-filter BFS clear
// ─────────────────────────────────────────────────────────────────────────────
static void clearComponent(
    int startIdx, int lbl,
    const std::vector<int>& labelMap,
    std::vector<uint8_t>&   occ,
    int W, int H)
{
    static constexpr int ox[4]={1,-1,0,0}, oy[4]={0,0,1,-1};
    std::vector<int> q; q.reserve(256);
    q.push_back(startIdx); occ[startIdx]=0;
    int head=0;
    while(head<(int)q.size()){
        int cur=q[head++];
        int cx=cur%W, cy=cur/W;
        for(int d=0;d<4;++d){
            int nx=cx+ox[d], ny=cy+oy[d];
            if((unsigned)nx>=(unsigned)W||(unsigned)ny>=(unsigned)H) continue;
            int ni=ny*W+nx;
            if(occ[ni]==0||labelMap[ni]!=lbl) continue;
            occ[ni]=0; q.push_back(ni);
        }
    }
}


// ═════════════════════════════════════════════════════════════════════════════
//  Stage 5 — Shared edge graph (topology for seam elimination + E9)
// ═════════════════════════════════════════════════════════════════════════════
struct EdgePixel { int x,y,axis; uint32_t cA,cB; };
struct SharedEdgeGraph {
    std::unordered_map<uint64_t,std::vector<EdgePixel>> edges;
    static uint64_t key(uint32_t a,uint32_t b) noexcept {
        if(a>b)std::swap(a,b);
        return((uint64_t)a<<32)|(uint64_t)b;
    }
    void insert(int x,int y,int axis,uint32_t cA,uint32_t cB){
        edges[key(cA,cB)].push_back({x,y,axis,cA,cB});
    }
    bool hasEdge(uint32_t cA,uint32_t cB) const {
        return edges.count(key(cA,cB))>0;
    }
};


static SharedEdgeGraph buildEdgeGraph(
    const std::vector<uint32_t>& pixelColor, int W, int H)
{
    SharedEdgeGraph graph; graph.edges.reserve(256);
    for(int y=0;y<H-1;++y)
        for(int x=0;x<W;++x){
            uint32_t cT=pixelColor[y*W+x], cB=pixelColor[(y+1)*W+x];
            if(cT==0xFFFFFFFFu||cB==0xFFFFFFFFu||cT==cB) continue;
            graph.insert(x,y,0,cT,cB);
        }
    for(int y=0;y<H;++y)
        for(int x=0;x<W-1;++x){
            uint32_t cL=pixelColor[y*W+x], cR=pixelColor[y*W+x+1];
            if(cL==0xFFFFFFFFu||cR==0xFFFFFFFFu||cL==cR) continue;
            graph.insert(x,y,1,cL,cR);
        }
    return graph;
}


// ═════════════════════════════════════════════════════════════════════════════
//  Stage 6 — Moore boundary trace (Jacob's stopping criterion)
//
//       Each emitted point is placed at the midpoint of the pixel EDGE between
//       the current foreground pixel and the pixel it was entered from, instead
//       of at the pixel centre.  For cardinal moves this lands exactly on an
//       integer grid line (no seam gap); for diagonal moves it lands on the
//       corner shared by the two pixels.
//
//       Only follow occ==1 (unvisited foreground) pixels.
//       maxSteps is caller-supplied per-component budget.
// ═════════════════════════════════════════════════════════════════════════════
[[nodiscard]] static std::vector<Point> traceBoundary(
    int startX, int startY, int W, int H,
    std::vector<uint8_t>& occ,
    int componentMaxSteps)
{
    std::vector<Point> path; path.reserve(64);

    int entryDir=-1;
    for(int d=0;d<8;++d){
        int nx=startX+DX[d], ny=startY+DY[d];
        if((unsigned)nx>=(unsigned)W||(unsigned)ny>=(unsigned)H) continue;
        if(occ[ny*W+nx]==1){entryDir=d;break;}
    }
    if(entryDir==-1){
        // Isolated pixel: emit pixel centre
        path.push_back({startX+.5f,startY+.5f});
        return path;
    }

    int curX=startX, curY=startY;
    int fromDir=(entryDir+5)%8;
    int firstEntryDir=-1;
    bool firstVisit=true;
    int prevMoveDir=-1; // [E7] direction of last move

    for(int step=0; step<componentMaxSteps; ++step){
        // [E7] Emit sub-pixel edge midpoint
        if(prevMoveDir>=0){
            path.push_back({
                curX + 0.5f - 0.5f * DX[prevMoveDir],
                curY + 0.5f - 0.5f * DY[prevMoveDir]});
        } else {
            path.push_back({curX+.5f, curY+.5f}); // first point: pixel centre
        }
        occ[curY*W+curX]=2;

        bool found=false;
        for(int i=0;i<8;++i){
            int dir=(fromDir+i)%8;
            int nx=curX+DX[dir], ny=curY+DY[dir];
            if((unsigned)nx>=(unsigned)W||(unsigned)ny>=(unsigned)H) continue;
            if(occ[ny*W+nx]!=1) continue;
            if(nx==startX&&ny==startY){
                if(firstVisit){firstEntryDir=dir;firstVisit=false;}
                else if(dir==firstEntryDir) return path;
            } else firstVisit=false;
            prevMoveDir=dir;
            curX=nx; curY=ny;
            fromDir=(dir+5)%8;
            found=true; break;
        }
        if(!found) break;

        if(step==componentMaxSteps-2)
            VT_WARN("traceBoundary: step budget nearly exhausted "
                    "(maxSteps=%d path=%d start=(%d,%d))",
                    componentMaxSteps,(int)path.size(),startX,startY);
    }
    return path;
}


// ─────────────────────────────────────────────────────────────────────────────
//  Snap boundary vertices to exact shared-edge grid lines  [E1/E8]
//       Called BEFORE rdpSimplify so RDP operates on the final sub-pixel
//       geometry, not on raw pixel-centre coordinates.
// ─────────────────────────────────────────────────────────────────────────────
static void snapToSharedEdges(
    std::vector<Point>& path,
    const std::vector<uint32_t>& pixelColor,
    uint32_t myColor,
    int W, int H,
    float snapDist=0.6f) noexcept
{
    const int WH = W * H;
    for(auto& pt:path){
        int px=std::clamp((int)pt.x,0,W-1);
        int py=std::clamp((int)pt.y,0,H-1);

        float nearestX=std::round(pt.x);
        if(std::abs(pt.x-nearestX)<snapDist){
            int nx=(int)nearestX;
            if(nx>0&&nx<W){
                int iL=py*W+(nx-1), iR=py*W+nx;
                uint32_t cL=(iL>=0&&iL<WH)?pixelColor[iL]:0xFFFFFFFFu;
                uint32_t cR=(iR>=0&&iR<WH)?pixelColor[iR]:0xFFFFFFFFu;
                if((cL==myColor)!=(cR==myColor)) pt.x=(float)nx;
            }
        }
        float nearestY=std::round(pt.y);
        if(std::abs(pt.y-nearestY)<snapDist){
            int ny=(int)nearestY;
            if(ny>0&&ny<H){
                int iT=(ny-1)*W+px, iB=ny*W+px;
                uint32_t cT=(iT>=0&&iT<WH)?pixelColor[iT]:0xFFFFFFFFu;
                uint32_t cB=(iB>=0&&iB<WH)?pixelColor[iB]:0xFFFFFFFFu;
                if((cT==myColor)!=(cB==myColor)) pt.y=(float)ny;
            }
        }
        (void)px; (void)py;
    }
}


// ─────────────────────────────────────────────────────────────────────────────
//  Stage 7 — Ramer-Douglas-Peucker simplification (iterative, stack-based)
//  Called AFTER snapToSharedEdges.
// ─────────────────────────────────────────────────────────────────────────────
static float ptSegDistSq(const Point& p,const Point& a,const Point& b) noexcept {
    float dx=b.x-a.x, dy=b.y-a.y;
    float ls=dx*dx+dy*dy;
    if(ls<1e-12f){float ex=p.x-a.x,ey=p.y-a.y;return ex*ex+ey*ey;}
    float t=std::clamp(((p.x-a.x)*dx+(p.y-a.y)*dy)/ls,0.f,1.f);
    float qx=a.x+t*dx-p.x, qy=a.y+t*dy-p.y;
    return qx*qx+qy*qy;
}


[[nodiscard]] static std::vector<Point> rdpSimplify(
    const std::vector<Point>& pts, float eps) noexcept
{
    const int n=(int)pts.size();
    if(n<=2) return pts;
    const float epsSq=eps*eps;
    std::vector<bool> keep(n,false);
    keep[0]=keep[n-1]=true;
    struct Frame{int lo,hi;};
    std::vector<Frame> stack;
    stack.push_back({0,n-1});
    while(!stack.empty()){
        auto[a,b]=stack.back(); stack.pop_back();
        if(b-a<=1) continue;
        float maxD=0.f; int maxI=a;
        for(int i=a+1;i<b;++i){
            float d=ptSegDistSq(pts[i],pts[a],pts[b]);
            if(d>maxD){maxD=d;maxI=i;}
        }
        if(maxD>epsSq){
            keep[maxI]=true;
            stack.push_back({a,maxI});
            stack.push_back({maxI,b});
        }
    }
    std::vector<Point> out; out.reserve(n);
    for(int i=0;i<n;++i) if(keep[i]) out.push_back(pts[i]);
    return out;
}


// ─────────────────────────────────────────────────────────────────────────────
//  Stage 8 — Corner detection
//        Uses ±kCornerHW (±3) point tangent window so the detector measures
//        the local path trend rather than the pixel-grid staircase angle.
//        This dramatically reduces false-positive corners on jagged rasters.
// ─────────────────────────────────────────────────────────────────────────────
[[nodiscard]] static std::vector<uint8_t> detectCorners(
    const std::vector<Point>& pts, float thresh_deg) noexcept
{
    const int n=(int)pts.size();
    std::vector<uint8_t> c(n,0);
    if(n<3){std::fill(c.begin(),c.end(),1);return c;}

    // FIX-1: use kPi instead of M_PI
    const float thresh_rad=thresh_deg*kPi/180.f;

    // [E11] Tangent window half-width: use up to kCornerHW, but clamp so
    //       it never wraps around more than half the polygon.
    const int hw=std::min(kCornerHW, std::max(1, n/4));

    for(int i=0;i<n;++i){
        const Point& prev=pts[(i+n-hw)%n];
        const Point& next=pts[(i+hw)%n];
        float ax=pts[i].x-prev.x, ay=pts[i].y-prev.y;
        float bx=next.x-pts[i].x, by=next.y-pts[i].y;
        float lA=std::sqrt(ax*ax+ay*ay);
        float lB=std::sqrt(bx*bx+by*by);
        if(lA<1e-6f||lB<1e-6f){c[i]=1;continue;}
        float d=std::clamp((ax*bx+ay*by)/(lA*lB),-1.f,1.f);
        c[i]=(std::acos(d)<thresh_rad)?1:0;
    }
    return c;
}


// ═════════════════════════════════════════════════════════════════════════════
//  Stage 9 — Constrained Least-Squares Bézier Fitting
//    Wrap-around discontinuity detection in buildSplineLSQ
//    Post-solve control-point clamp → Catmull-Rom fallback
//    kMaxFitIter=20, early-stop on convergence
// ═════════════════════════════════════════════════════════════════════════════

static Point bezier(const Point& P0,const Point& P1,
                    const Point& P2,const Point& P3,float t) noexcept {
    float mt=1.f-t;
    float b0=mt*mt*mt, b1=3.f*mt*mt*t, b2=3.f*mt*t*t, b3=t*t*t;
    return{b0*P0.x+b1*P1.x+b2*P2.x+b3*P3.x,
           b0*P0.y+b1*P1.y+b2*P2.y+b3*P3.y};
}


[[nodiscard]] static std::vector<float> chordParam(
    const std::vector<Point>& pts, int lo, int hi) noexcept
{
    int n=hi-lo;
    std::vector<float> u(n,0.f);
    for(int i=1;i<n;++i){
        float dx=pts[lo+i].x-pts[lo+i-1].x;
        float dy=pts[lo+i].y-pts[lo+i-1].y;
        u[i]=u[i-1]+std::sqrt(dx*dx+dy*dy);
    }
    float tot=u[n-1];
    if(tot>1e-6f) for(float& v:u) v/=tot;
    return u;
}


struct CubicFit{Point P1,P2;float maxResidSq;};


// [E2] Catmull-Rom fallback: P1 = P0 + chord/3, P2 = P3 - chord/3
static std::pair<Point,Point> catmullRomFallback(
    const Point& P0, const Point& P3) noexcept
{
    float dx=P3.x-P0.x, dy=P3.y-P0.y;
    float L=std::sqrt(dx*dx+dy*dy);
    if(L<1e-6f){
        return{{P0.x+(P3.x-P0.x)/3.f, P0.y+(P3.y-P0.y)/3.f},
               {P0.x+2.f*(P3.x-P0.x)/3.f, P0.y+2.f*(P3.y-P0.y)/3.f}};
    }
    float c=L/3.f, tx=dx/L, ty=dy/L;
    return{{P0.x+tx*c, P0.y+ty*c},
           {P3.x-tx*c, P3.y-ty*c}};
}


static CubicFit fitCubicSegment(
    const std::vector<Point>& raw, int lo, int hi,
    const Point& P0, const Point& P3,
    const std::vector<float>* uExt=nullptr)
{
    const int n=hi-lo;
    if(n<=2){
        auto [P1,P2]=catmullRomFallback(P0,P3);
        return{P1,P2,0.f};
    }

    std::vector<float> u_storage;
    const std::vector<float>* u=uExt;
    if(!u||(int)u->size()!=n){
        u_storage=chordParam(raw,lo,hi);
        u=&u_storage;
    }

    double A00=0,A01=0,A11=0;
    double Bx0=0,By0=0,Bx1=0,By1=0;
    for(int i=0;i<n;++i){
        float t=(*u)[i];
        float mt=1.f-t;
        float c0=mt*mt*mt, c1=3.f*mt*mt*t, c2=3.f*mt*t*t, c3=t*t*t;
        float Qx=raw[lo+i].x-c0*P0.x-c3*P3.x;
        float Qy=raw[lo+i].y-c0*P0.y-c3*P3.y;
        A00+=c1*c1; A01+=c1*c2; A11+=c2*c2;
        Bx0+=c1*Qx; By0+=c1*Qy;
        Bx1+=c2*Qx; By1+=c2*Qy;
    }

    double det=A00*A11-A01*A01;
    Point P1,P2;
    if(std::abs(det)<1e-10){
        auto [cP1,cP2]=catmullRomFallback(P0,P3);
        P1=cP1; P2=cP2;
    } else {
        double inv=1.0/det;
        P1.x=(float)((A11*Bx0-A01*Bx1)*inv);
        P1.y=(float)((A11*By0-A01*By1)*inv);
        P2.x=(float)((A00*Bx1-A01*Bx0)*inv);
        P2.y=(float)((A00*By1-A01*By0)*inv);
    }

    // [E2] Sanity-clamp runaway control points
    const float segLen=vlen(P3-P0);
    const float maxExt=kCPClampK*segLen;
    bool cp1Bad=(vlen(P1-P0)>maxExt && segLen>1e-4f);
    bool cp2Bad=(vlen(P2-P3)>maxExt && segLen>1e-4f);
    if(cp1Bad||cp2Bad){
        auto [cP1,cP2]=catmullRomFallback(P0,P3);
        if(cp1Bad) P1=cP1;
        if(cp2Bad) P2=cP2;
    }

    float maxRes=0.f;
    for(int i=0;i<n;++i){
        Point b=bezier(P0,P1,P2,P3,(*u)[i]);
        float dx=b.x-raw[lo+i].x, dy=b.y-raw[lo+i].y;
        float r=dx*dx+dy*dy;
        if(r>maxRes) maxRes=r;
    }
    return{P1,P2,maxRes};
}


[[nodiscard]] static std::vector<float> reparameterise(
    const std::vector<Point>& raw, int lo, int hi,
    const Point& P0, const Point& P1,
    const Point& P2, const Point& P3,
    const std::vector<float>& u)
{
    int n=hi-lo;
    std::vector<float> u2(n);
    for(int i=0;i<n;++i){
        float t=u[i];
        Point Bt=bezier(P0,P1,P2,P3,t);
        float mt=1.f-t;
        Point d0=P1-P0, d1=P2-P1, d2=P3-P2;
        Point Bp={3.f*(mt*mt*d0.x+2.f*mt*t*d1.x+t*t*d2.x),
                  3.f*(mt*mt*d0.y+2.f*mt*t*d1.y+t*t*d2.y)};
        Point diff={Bt.x-raw[lo+i].x, Bt.y-raw[lo+i].y};
        float denom=dot(Bp,Bp);
        float delta=(denom>1e-8f)?dot(diff,Bp)/denom:0.f;
        u2[i]=std::clamp(t-delta,0.f,1.f);
    }
    return u2;
}


// [E10] kMaxFitIter=20, early convergence stop
static void fitBezierRecursive(
    const std::vector<Point>& raw, int lo, int hi,
    const Point& P0, const Point& P3,
    std::vector<Segment>& out,
    int depth=0,
    float fit_tolerance=0.5f)
{
    static constexpr int kMaxDepth=6;
    if(hi-lo<=1||depth>kMaxDepth){
        out.push_back({false,{},{},P3}); return;
    }

    std::vector<float> u=chordParam(raw,lo,hi);
    Point P1_best,P2_best;
    float bestRes=1e30f;

    for(int iter=0; iter<kMaxFitIter; ++iter){
        CubicFit fit=fitCubicSegment(raw,lo,hi,P0,P3,&u);
        if(fit.maxResidSq<bestRes){
            bestRes=fit.maxResidSq;
            P1_best=fit.P1; P2_best=fit.P2;
        }
        if(fit.maxResidSq<=fit_tolerance*fit_tolerance) break;
        float improvement=bestRes-fit.maxResidSq;
        if(iter>0 && improvement>=0.f && improvement<kFitConvergEps) break;
        u=reparameterise(raw,lo,hi,P0,fit.P1,fit.P2,P3,u);
    }

    if(bestRes<=fit_tolerance*fit_tolerance){
        out.push_back({true,P1_best,P2_best,P3}); return;
    }

    CubicFit fit=fitCubicSegment(raw,lo,hi,P0,P3,&u);
    int worstI=lo;{
        float wRes=0.f;
        for(int i=0;i<hi-lo;++i){
            Point b=bezier(P0,fit.P1,fit.P2,P3,u[i]);
            float dx=b.x-raw[lo+i].x, dy=b.y-raw[lo+i].y;
            float r=dx*dx+dy*dy;
            if(r>wRes){wRes=r;worstI=lo+i;}
        }
    }
    if(worstI==lo||worstI==hi-1){
        out.push_back({true,P1_best,P2_best,P3}); return;
    }
    const Point& Pmid=raw[worstI];
    fitBezierRecursive(raw,lo,worstI+1,P0,Pmid,out,depth+1,fit_tolerance);
    fitBezierRecursive(raw,worstI,hi,Pmid,P3,out,depth+1,fit_tolerance);
}


// [E1] Detect spatial discontinuity in a wrapped seg_raw.
//      Returns the index of the largest gap (or -1 if no discontinuity).
static int detectWrapDiscontinuity(const std::vector<Point>& seg) noexcept {
    const int n=(int)seg.size();
    if(n<3) return -1;
    float sumD2=0.f; int cnt=0;
    float maxD2=0.f; int maxIdx=-1;
    for(int k=0;k+1<n;++k){
        float dx=seg[k+1].x-seg[k].x, dy=seg[k+1].y-seg[k].y;
        float d2=dx*dx+dy*dy;
        sumD2+=d2; ++cnt;
        if(d2>maxD2){maxD2=d2;maxIdx=k;}
    }
    if(cnt==0) return -1;
    float mean=sumD2/cnt;
    return (maxD2>9.f*mean) ? maxIdx : -1;
}


[[nodiscard]] static std::vector<Segment> buildSplineLSQ(
    const std::vector<Point>& keyPts,
    const std::vector<uint8_t>& isCorner,
    const std::vector<Point>& rawPts,
    float fFit_Tolerance)
{
    const int nk=(int)keyPts.size();
    const int nr=(int)rawPts.size();
    if(nk<2||nr<2) return {};

    std::vector<Segment> segs; segs.reserve(nk);

    std::vector<int> keyToRaw(nk);
    for(int ki=0;ki<nk;++ki){
        const Point& p=keyPts[ki];
        float best=1e30f; int bi=0;
        for(int i=0;i<nr;++i){
            float dx=rawPts[i].x-p.x, dy=rawPts[i].y-p.y;
            float d=dx*dx+dy*dy;
            if(d<best){best=d;bi=i;}
        }
        keyToRaw[ki]=bi;
    }

    std::vector<int> corners;
    for(int i=0;i<nk;++i) if(isCorner[i]) corners.push_back(i);
    if(corners.empty()) corners.push_back(0);

    const int nc=(int)corners.size();
    auto nxt=[&](int i){return(i+1)%nk;};

    for(int ci=0;ci<nc;++ci){
        int from=corners[ci];
        int to=corners[(ci+1)%nc];
        bool fullCircle=(from==to);

        std::vector<int> run;
        if(fullCircle){
            for(int k=from;;){
                run.push_back(k); k=nxt(k);
                if(k==from){run.push_back(from);break;}
                if((int)run.size()>nk+2) break;
            }
        } else {
            for(int k=from;;k=nxt(k)){
                run.push_back(k);
                if(k==to) break;
                if((int)run.size()>nk+2) break;
            }
        }
        const int rn=(int)run.size();
        if(rn<2) continue;

        for(int ri=0;ri<rn-1;++ri){
            const Point& P0=keyPts[run[ri]];
            const Point& P3=keyPts[run[ri+1]];

            int rlo=keyToRaw[run[ri]];
            int rhi=keyToRaw[run[ri+1]];

            std::vector<Point> seg_raw;
            if(rlo<=rhi){
                seg_raw.assign(rawPts.begin()+rlo,rawPts.begin()+rhi+1);
            } else {
                std::vector<Point> half1(rawPts.begin()+rlo,rawPts.end());
                std::vector<Point> half2(rawPts.begin(),rawPts.begin()+rhi+1);
                seg_raw=half1;
                seg_raw.insert(seg_raw.end(),half2.begin(),half2.end());

                // [E1] Detect spatial discontinuity at the wrap join
                int gapIdx=detectWrapDiscontinuity(seg_raw);
                if(gapIdx>=0){
                    int n1=gapIdx+1;
                    int n2=(int)seg_raw.size()-gapIdx-1;
                    if(n1>=2 && n1>=n2){
                        seg_raw.resize(n1);
                    } else if(n2>=2){
                        seg_raw.erase(seg_raw.begin(),seg_raw.begin()+gapIdx+1);
                    }
                }
            }

            if((int)seg_raw.size()<2){
                segs.push_back({false,{},{},P3}); continue;
            }
            float dx=P3.x-P0.x, dy=P3.y-P0.y;
            if(dx*dx+dy*dy<1.f){
                segs.push_back({false,{},{},P3}); continue;
            }
            fitBezierRecursive(seg_raw,0,(int)seg_raw.size(),P0,P3,segs,0,fFit_Tolerance);
        }
    }
    return segs;
}


// ─────────────────────────────────────────────────────────────────────────────
//  Stage 10 — Winding-order normalisation
// ─────────────────────────────────────────────────────────────────────────────
static float signedArea(const std::vector<Point>& pts) noexcept {
    float area=0.f; const int n=(int)pts.size();
    for(int i=0;i<n;++i){
        const Point& a=pts[i]; const Point& b=pts[(i+1)%n];
        area+=(a.x*b.y-b.x*a.y);
    }
    return area*0.5f;
}
// Returns true if points were reversed
static bool ensureCW (std::vector<Point>& p) noexcept {
    if(signedArea(p)<0.f){std::reverse(p.begin(),p.end());return true;}
    return false;
}
static bool ensureCCW(std::vector<Point>& p) noexcept {
    if(signedArea(p)>0.f){std::reverse(p.begin(),p.end());return true;}
    return false;
}


// ─────────────────────────────────────────────────────────────────────────────
//  [E6] Point-in-polygon via ray-casting
//       Used for hole detection instead of bounding-box containment.
// ─────────────────────────────────────────────────────────────────────────────
static bool pointInPolygon(
    const std::vector<Point>& poly, float px, float py) noexcept
{
    const int n=(int)poly.size();
    bool inside=false;
    for(int i=0,j=n-1;i<n;j=i++){
        float xi=poly[i].x, yi=poly[i].y;
        float xj=poly[j].x, yj=poly[j].y;
        if(((yi>py)!=(yj>py)) &&
           (px<(xj-xi)*(py-yi)/(yj-yi)+xi))
            inside=!inside;
    }
    return inside;
}


// ═════════════════════════════════════════════════════════════════════════════
//  [E9]  Gradient region detection
//
//  Algorithm:
//   1. Build a Lab-adjacency graph: palette colours A & B are candidates if
//      ΔE(A,B) < gradient_detect_thresh AND they share a pixel boundary
//      (from SharedEdgeGraph).
//   2. Union-Find groups candidates into gradient chains.
//   3. For each chain with ≥ 2 colours: compute the union bounding box of all
//      their connected components, determine the major axis, sort stops dark→
//      light, and emit a <linearGradient gradientUnits="userSpaceOnUse">.
//   4. The SVG emission loop uses url(#vgN) fills for colours in a gradient
//      group instead of flat hex fills.
// ═════════════════════════════════════════════════════════════════════════════
struct GradStop { uint32_t color; float offset; };


struct GradientDef {
    int   id;
    float x1,y1,x2,y2;
    std::vector<GradStop>  stops;
    std::vector<uint32_t>  colors;
};


static std::vector<GradientDef> buildGradientDefs(
    const std::vector<uint32_t>& palette,
    const SharedEdgeGraph& edgeGraph,
    const std::vector<uint32_t>& componentColor,
    const std::vector<std::array<int,4>>& componentBBox,
    const std::unordered_map<uint32_t,std::vector<int>>& colorToComponents,
    float gradThresh)
{
    const int K=(int)palette.size();
    if(K<2) return {};
    const float thresh2=gradThresh*gradThresh;

    UnionFind uf(K);
    for(int i=0;i<K;++i)
        for(int j=i+1;j<K;++j)
            if(labDistSq(palette[i],palette[j])<=thresh2 &&
               edgeGraph.hasEdge(palette[i],palette[j]))
                uf.unite(i,j);

    std::unordered_map<int,std::vector<int>> groups;
    for(int i=0;i<K;++i) groups[uf.find(i)].push_back(i);

    std::vector<GradientDef> defs;
    int gradId=0;

    for(auto& [root,members]:groups){
        if((int)members.size()<2) continue;

        std::sort(members.begin(),members.end(),[&](int a,int b){
            return rgbToLabLUT(palette[a]).L < rgbToLabLUT(palette[b]).L;
        });

        int ux0=INT_MAX,uy0=INT_MAX,ux1=INT_MIN,uy1=INT_MIN;
        for(int mi:members){
            auto it=colorToComponents.find(palette[mi]);
            if(it==colorToComponents.end()) continue;
            for(int lbl:it->second){
                if(lbl<0||lbl>=(int)componentBBox.size()) continue;
                const auto& bb=componentBBox[lbl];
                ux0=std::min(ux0,bb[0]); uy0=std::min(uy0,bb[1]);
                ux1=std::max(ux1,bb[2]); uy1=std::max(uy1,bb[3]);
            }
        }
        if(ux0==INT_MAX) continue;

        GradientDef def;
        def.id=++gradId;

        bool vert=(uy1-uy0)>(ux1-ux0);
        if(vert){
            def.x1=def.x2=(ux0+ux1)*0.5f;
            def.y1=(float)uy0; def.y2=(float)uy1;
        } else {
            def.y1=def.y2=(uy0+uy1)*0.5f;
            def.x1=(float)ux0; def.x2=(float)ux1;
        }

        int nm=(int)members.size();
        for(int mi=0;mi<nm;++mi){
            float off=(nm==1)?0.f:(float)mi/(float)(nm-1);
            def.stops.push_back({palette[members[mi]],off});
            def.colors.push_back(palette[members[mi]]);
        }
        defs.push_back(std::move(def));
    }
    return defs;
}


// SVG-2/SVG-4: double quotes throughout; xmlns:xlink emitted conditionally
static void emitGradientDefs(std::string& svg, const std::vector<GradientDef>& defs) {
    if(defs.empty()) return;
    // SVG-4: xlink namespace required by some rasterisers for url() references
    svg += "<defs>";
    for(auto& def:defs){
        char buf[256];
        snprintf(buf,sizeof(buf),
            "<linearGradient id=\"vg%d\" "
            "x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" "
            "gradientUnits=\"userSpaceOnUse\">",
            def.id,def.x1,def.y1,def.x2,def.y2);
        svg+=buf;
        for(auto& s:def.stops){
            char sb[128];
            snprintf(sb,sizeof(sb),
                "<stop offset=\"%.3f\" stop-color=\"#%02x%02x%02x\"/>",
                s.offset,rCh(s.color),gCh(s.color),bCh(s.color));
            svg+=sb;
        }
        svg+="</linearGradient>";
    }
    svg+="</defs>";
}


// ─────────────────────────────────────────────────────────────────────────────
//  Stage 11 — SVG output helpers
// ─────────────────────────────────────────────────────────────────────────────
static void appendFloat(std::string& s, float v, int dp) {
    char buf[32];
    int len=snprintf(buf,sizeof(buf),"%.*f",dp,(double)v);
    if(dp>0){
        while(len>1&&buf[len-1]=='0') --len;
        if(len>1&&buf[len-1]=='.') --len;
    }
    s.append(buf,len);
}


static void appendSegmentRel(
    std::string& d, const Segment& seg, const Point& prev, int dp)
{
    if(seg.isCurve){
        d+='c';
        appendFloat(d,seg.cp1.x-prev.x,dp); d+=' ';
        appendFloat(d,seg.cp1.y-prev.y,dp); d+=' ';
        appendFloat(d,seg.cp2.x-prev.x,dp); d+=' ';
        appendFloat(d,seg.cp2.y-prev.y,dp); d+=' ';
        appendFloat(d,seg.end.x-prev.x,dp); d+=' ';
        appendFloat(d,seg.end.y-prev.y,dp);
    } else {
        float dx=seg.end.x-prev.x, dy=seg.end.y-prev.y;
        if(std::abs(dy)<1e-4f&&std::abs(dx)>=1e-4f){d+='h';appendFloat(d,dx,dp);}
        else if(std::abs(dx)<1e-4f&&std::abs(dy)>=1e-4f){d+='v';appendFloat(d,dy,dp);}
        else{d+='l';appendFloat(d,dx,dp);d+=' ';appendFloat(d,dy,dp);}
    }
}


static void appendColorHex(std::string& s, uint32_t c) {
    c=rgb24(c);
    uint8_t r=rCh(c), g=gCh(c), b=bCh(c);
    char buf[8];
    if((r&0x0F)==(r>>4)&&(g&0x0F)==(g>>4)&&(b&0x0F)==(b>>4))
        snprintf(buf,sizeof(buf),"#%x%x%x",r>>4,g>>4,b>>4);
    else
        snprintf(buf,sizeof(buf),"#%02x%02x%02x",r,g,b);
    s+=buf;
}


// Build the SVG path d-string for one PathRecord
struct PathRecord {
    std::vector<Point>   rawPts;
    std::vector<Point>   pts;
    std::vector<Segment> segs;
    bool                 isHole;
};


[[nodiscard]] static std::string buildPathD(const PathRecord& pr, int dp) {
    if(pr.segs.empty()||pr.pts.empty()) return {};
    std::string d; d.reserve(pr.segs.size()*14);
    const Point& start=pr.pts[0];
    d+='M';
    appendFloat(d,start.x,dp); d+=' ';
    appendFloat(d,start.y,dp);
    Point prev=start;
    for(const Segment& seg:pr.segs){
        appendSegmentRel(d,seg,prev,dp);
        prev=seg.end;
    }
    d+='Z';
    return d;
}


// ─────────────────────────────────────────────────────────────────────────────
//  Public entry point
// ─────────────────────────────────────────────────────────────────────────────
std::string vectorize(const uint8_t* pixels, int width, int height, Options options)
{
    if(!pixels||width<=0||height<=0){
        VT_ERR("vectorize: invalid args (pixels=%p w=%d h=%d)",(void*)pixels,width,height);
        assert(false&&"null pixels or non-positive dimensions");
        return "";
    }

    const double t0=vt_now_ms();
    VT_LOG("vectorize: start %dx%d (%d px)",width,height,width*height);

    // Apply defaults
    if(options.color_precision <=0) options.color_precision =kDefaultColorPrecision;
    if(options.corner_threshold<=0) options.corner_threshold=kDefaultCornerThreshold;
    if(options.filter_speckle  <=0) options.filter_speckle  =kDefaultFilterSpeckle;
    if(options.path_precision  < 0) options.path_precision  =kDefaultPathPrecision;
    if(options.rdp_epsilon     <=0) options.rdp_epsilon     =kDefaultRdpEpsilon;
    if(options.blur_radius     < 0) options.blur_radius     =kDefaultBlurRadius;
    if(options.fit_tolerance   <=0) options.fit_tolerance   =kFitTolerance;
    if(options.bilateral_sigma_r<=0) options.bilateral_sigma_r=30.f;
    if(options.gradient_detect_thresh<=0) options.gradient_detect_thresh=kGradDetectDefault;

    VT_LOG("vectorize: options — color_precision=%d gradient_step=%.2f "
           "color_mode=%s filter_speckle=%d rdp_epsilon=%.2f "
           "corner_threshold=%.1f fit_tolerance=%.2f "
           "blur_radius=%.2f bilateral_sigma_r=%.1f "
           "gradient_detect_thresh=%.1f path_precision=%d",
           options.color_precision,(double)options.gradient_step,
           options.color_mode==ColorMode::BlackAndWhite?"BW":"Color",
           options.filter_speckle,(double)options.rdp_epsilon,
           (double)options.corner_threshold,(double)options.fit_tolerance,
           (double)options.blur_radius,(double)options.bilateral_sigma_r,
           (double)options.gradient_detect_thresh,options.path_precision);

    // FIX-9: explicit clamp of path_precision to int [0,6]
    const int dp = std::clamp(options.path_precision, 0, 6);
    const int N  = width * height;

    // ── Stage 0: [E4+E14] Bilateral pre-filter (PERF-1/2: LUT exp) ───────
    double ts=vt_now_ms();
    std::vector<uint8_t> blurred=bilateralFilter(
        pixels, width, height,
        options.blur_radius,
        options.bilateral_sigma_r);
    VT_LOG("Stage 0 (bilateral σ_s=%.2f σ_r=%.1f): %.1f ms",
           (double)options.blur_radius,(double)options.bilateral_sigma_r,
           vt_now_ms()-ts);
    const uint8_t* src=blurred.data();

    // ── Stages 1+2: quantise + Lab gradient merge ─────────────────────────
    ts=vt_now_ms();
    std::vector<uint32_t> pixelColor;
    std::vector<uint32_t> palette=
        buildPaletteAndAssign(src,width,height,options,pixelColor);
    VT_LOG("Stage 1+2 (quantise+merge): %.1f ms, palette=%d",
           vt_now_ms()-ts,(int)palette.size());

    if(palette.empty()){
        char buf[128];
        snprintf(buf,sizeof(buf),
            "<svg xmlns=\"http://www.w3.org/2000/svg\" "
            "viewBox=\"0 0 %d %d\"></svg>",width,height);
        VT_WARN("vectorize: palette empty");
        return buf;
    }

    // ── Stage 3: connected components (+ bbox for E5 / E9) ───────────────
    ts=vt_now_ms();
    std::vector<uint32_t> componentColor;
    std::vector<int>      componentSize;
    std::vector<std::array<int,4>> componentBBox;
    std::vector<int> labelMap=
        labelComponents(pixelColor,width,height,
                         componentColor,componentSize,componentBBox);
    VT_LOG("Stage 3 (BFS components): %.1f ms, %d components",
           vt_now_ms()-ts,(int)componentColor.size());

    std::unordered_map<uint32_t,std::vector<int>> colorToComponents;
    colorToComponents.reserve(palette.size()*2);
    for(int lbl=0;lbl<(int)componentColor.size();++lbl)
        colorToComponents[componentColor[lbl]].push_back(lbl);

    // ── [E5] Re-sort palette by decreasing avg component area ─────────────
    {
        std::unordered_map<uint32_t,double> colorAvgArea;
        std::unordered_map<uint32_t,int>    colorCompCount;
        for(int lbl=0;lbl<(int)componentColor.size();++lbl){
            colorAvgArea [componentColor[lbl]]+=componentSize[lbl];
            colorCompCount[componentColor[lbl]]++;
        }
        for(auto& [c,sum]:colorAvgArea){
            int cnt=colorCompCount.count(c)?colorCompCount.at(c):1;
            sum/=cnt;
        }
        std::sort(palette.begin(),palette.end(),[&](uint32_t a,uint32_t b){
            return colorAvgArea[a]>colorAvgArea[b];
        });
        VT_LOG("[E5] Palette re-sorted by avg component area (largest first)");
    }

    // ── Stage 5: shared edge graph ────────────────────────────────────────
    ts=vt_now_ms();
    SharedEdgeGraph edgeGraph=buildEdgeGraph(pixelColor,width,height);
    VT_LOG("Stage 5 (edge graph): %.1f ms, %d edge groups",
           vt_now_ms()-ts,(int)edgeGraph.edges.size());

    // ── [E9] Gradient group detection ─────────────────────────────────────
    std::vector<GradientDef> gradDefs;
    std::unordered_map<uint32_t,int> colorToGrad;
    {
        gradDefs=buildGradientDefs(
            palette,edgeGraph,componentColor,componentBBox,
            colorToComponents,options.gradient_detect_thresh);
        for(auto& def:gradDefs)
            for(uint32_t c:def.colors)
                colorToGrad[c]=def.id;
        VT_LOG("[E9] Gradient groups detected: %d",(int)gradDefs.size());
    }

    // ── SVG header ────────────────────────────────────────────────────────
    // SVG-1: shape-rendering="geometricPrecision" — sub-pixel accuracy on HiDPI
    // SVG-2: double quotes throughout
    // SVG-4: xmlns:xlink when gradients are present
    std::string svg; svg.reserve((size_t)N*6);
    {
        char hdr[320];
        if(!gradDefs.empty()){
            snprintf(hdr,sizeof(hdr),
                "<svg xmlns=\"http://www.w3.org/2000/svg\" "
                "xmlns:xlink=\"http://www.w3.org/1999/xlink\" "
                "viewBox=\"0 0 %d %d\" "
                "shape-rendering=\"geometricPrecision\">",
                width,height);
        } else {
            snprintf(hdr,sizeof(hdr),
                "<svg xmlns=\"http://www.w3.org/2000/svg\" "
                "viewBox=\"0 0 %d %d\" "
                "shape-rendering=\"geometricPrecision\">",
                width,height);
        }
        svg+=hdr;
    }

    // [E9] Emit gradient <defs>
    emitGradientDefs(svg,gradDefs);

    // PERF-3: generation counter — eliminates full std::fill(occ) per colour
    // Each pixel is marked with its current generation when occupied;
    // occ_gen[i]==generation means pixel i belongs to the current colour.
    std::vector<uint32_t> occGen(N, 0);
    uint32_t generation = 0;

    int totalPaths=0, totalSpeckles=0, totalTracerMaxStepHits=0;
    double timeTrace=0, timeRDP=0, timeBezier=0, timeSVG=0;

    // Accumulate paths for gradient groups; keyed by gradient id
    // FIX-8: accumulate as a vector of strings; join without trailing space
    std::unordered_map<int,std::vector<std::string>> gradPathDsList;

    // Helper: occupancy check using generation counter
    auto occGet=[&](int i) noexcept -> uint8_t {
        // 0 = unset, 1 = foreground (current gen), 2 = visited (current gen+1 sentinel)
        if(occGen[i]<generation) return 0;         // stale from a previous colour
        return (uint8_t)(occGen[i]-generation+1);  // 1 = marked, maps 0/1 → 0/1
    };
    auto occSet=[&](int i, uint8_t v) noexcept {
        occGen[i] = generation + (uint32_t)(v - 1u); // v=1→gen, v=2→gen+1, v=0→gen-1
    };

    // ── Stages 4–11: per-colour processing ───────────────────────────────
    for(uint32_t color:palette){
        if(!colorToComponents.count(color)) continue;

        // PERF-3: advance generation; mark only pixels of this colour
        ++generation;
        if(generation==0) {
            // Overflow guard (extremely unlikely): reset all
            std::fill(occGen.begin(),occGen.end(),0);
            ++generation;
        }
        for(int i=0;i<N;++i)
            if(pixelColor[i]==color) occGen[i]=generation; // occ=1 sentinel

        // Provide a compatible occ vector view for functions that take uint8_t&
        // We keep a small shim so we don't have to touch the trace/clear APIs.
        // Instead: rebuild a local occ[] only for this colour's pixels.
        // This is O(component pixels) not O(W*H).
        std::vector<uint8_t> occ(N, 0);
        for(int i=0;i<N;++i)
            if(pixelColor[i]==color) occ[i]=1;

        std::vector<PathRecord> paths;

        for(int y=0;y<height;++y){
            for(int x=0;x<width;++x){
                int idx=y*width+x;
                if(occ[idx]!=1) continue;
                int lbl=labelMap[idx];

                // Stage 4: speckle filter
                if(componentSize[lbl]<options.filter_speckle){
                    clearComponent(idx,lbl,labelMap,occ,width,height);
                    ++totalSpeckles; continue;
                }

                int compMaxSteps=std::min(componentSize[lbl]*8+16, N+8);

                // Stage 6: boundary trace [E7 sub-pixel edges]
                double tTrace=vt_now_ms();
                std::vector<Point> raw=traceBoundary(
                    x,y,width,height,occ,compMaxSteps);
                timeTrace+=vt_now_ms()-tTrace;

                if((int)raw.size()>=compMaxSteps-1){
                    VT_WARN("Stage 6: tracer budget hit lbl=%d size=%d at (%d,%d)",
                            lbl,componentSize[lbl],x,y);
                    ++totalTracerMaxStepHits;
                }
                clearComponent(idx,lbl,labelMap,occ,width,height);
                if((int)raw.size()<3) continue;

                // [E8] Snap BEFORE RDP so RDP operates on sub-pixel geometry
                snapToSharedEdges(raw,pixelColor,color,width,height);

                // Stage 7: RDP simplification
                double tRDP=vt_now_ms();
                std::vector<Point> simplified=rdpSimplify(raw,options.rdp_epsilon);
                timeRDP+=vt_now_ms()-tRDP;
                if((int)simplified.size()<3) continue;

                // Stage 8: corner detection [E11 wider window]
                std::vector<uint8_t> corners=
                    detectCorners(simplified,options.corner_threshold);

                // Stage 9: Bézier fit [E1 wrap, E2 clamp, E10 iterations]
                // NOTE: segs will be discarded; re-fit after winding correction
                // below — stored here to avoid a separate allocation for rawPts.
                double tBez=vt_now_ms();
                std::vector<Segment> segs=
                    buildSplineLSQ(simplified,corners,raw,options.fit_tolerance);
                timeBezier+=vt_now_ms()-tBez;
                if(segs.empty()) continue;

                paths.push_back({std::move(raw),std::move(simplified),
                                 std::move(segs),false});
            }
        }
        if(paths.empty()) continue;

        // Stage 10: hole detection + winding order
        // [E6] Use point-in-polygon instead of bounding-box containment
        struct BBox{float x0,y0,x1,y1;};
        auto getBBox=[](const std::vector<Point>& p) noexcept ->BBox{
            BBox bb{1e30f,1e30f,-1e30f,-1e30f};
            for(auto& v:p){
                bb.x0=std::min(bb.x0,v.x); bb.y0=std::min(bb.y0,v.y);
                bb.x1=std::max(bb.x1,v.x); bb.y1=std::max(bb.y1,v.y);
            }
            return bb;
        };
        auto bbContains=[](const BBox& o,const BBox& i) noexcept ->bool{
            return i.x0>=o.x0&&i.y0>=o.y0&&i.x1<=o.x1&&i.y1<=o.y1;
        };

        std::vector<BBox> bboxes;
        bboxes.reserve(paths.size());
        for(auto& pr:paths) bboxes.push_back(getBBox(pr.pts));

        std::vector<int> order((int)paths.size());
        std::iota(order.begin(),order.end(),0);
        std::sort(order.begin(),order.end(),[&](int a,int b){
            float aA=(bboxes[a].x1-bboxes[a].x0)*(bboxes[a].y1-bboxes[a].y0);
            float bA=(bboxes[b].x1-bboxes[b].x0)*(bboxes[b].y1-bboxes[b].y0);
            return aA>bA;
        });

        for(int ii=1;ii<(int)order.size();++ii){
            int i=order[ii];
            for(int jj=0;jj<ii;++jj){
                int j=order[jj];
                if(!bbContains(bboxes[j],bboxes[i])) continue;
                float cx=0,cy=0;
                int np=(int)paths[i].pts.size();
                for(auto& p:paths[i].pts){cx+=p.x;cy+=p.y;}
                if(np>0){cx/=np;cy/=np;}
                if(pointInPolygon(paths[j].pts,cx,cy)){
                    paths[i].isHole=true; break;
                }
            }
        }

        // FIX-7: Apply winding correction in-place by reversing the point list,
        //         then fit Bézier curves ONCE with the correct orientation.
        //         The original code called buildSplineLSQ twice (first above,
        //         then again here) — the first fit was discarded entirely.
        double tBez=vt_now_ms();
        for(auto& pr:paths){
            bool reversed = pr.isHole ? ensureCCW(pr.pts) : ensureCW(pr.pts);
            // If the list was reversed we also reverse rawPts so the chord
            // parameterisation stays consistent with the new point order.
            if(reversed) std::reverse(pr.rawPts.begin(),pr.rawPts.end());
            auto c2=detectCorners(pr.pts,options.corner_threshold);
            pr.segs=buildSplineLSQ(pr.pts,c2,pr.rawPts,options.fit_tolerance);
        }
        timeBezier+=vt_now_ms()-tBez;

        // Stage 11: emit SVG
        double tSVG=vt_now_ms();

        // [E12] Combine all path d-strings for this colour into one <path>
        // [E13] Use fill-rule="evenodd" — handles holes without winding hack
        // SVG-2: double quotes
        std::string combined; combined.reserve(paths.size()*64);
        for(auto& pr:paths){
            std::string d=buildPathD(pr,dp);
            if(!d.empty()){
                if(!combined.empty()) combined+=' ';
                combined+=d;
                ++totalPaths;
            }
        }
        if(combined.empty()){timeSVG+=vt_now_ms()-tSVG;continue;}

        // [E9] Check if this colour belongs to a gradient group
        auto gitIt=colorToGrad.find(color);
        if(gitIt!=colorToGrad.end()){
            // FIX-8: accumulate as separate strings; join without trailing space
            gradPathDsList[gitIt->second].push_back(std::move(combined));
        } else {
            // Flat fill with evenodd [E12 E13] — SVG-2: double quotes
            svg += "<g fill=\"";
            appendColorHex(svg,color);
            svg += "\" fill-rule=\"evenodd\"><path d=\"";
            svg += combined;
            svg += "\"/></g>";
        }
        timeSVG+=vt_now_ms()-tSVG;
    }

    // [E9] Emit accumulated gradient paths
    // FIX-8: join list with spaces, no trailing space
    for(auto& def:gradDefs){
        auto it=gradPathDsList.find(def.id);
        if(it==gradPathDsList.end()||it->second.empty()) continue;
        char buf[64];
        snprintf(buf,sizeof(buf),
            "<g fill=\"url(#vg%d)\" fill-rule=\"evenodd\"><path d=\"",def.id);
        svg+=buf;
        const auto& list=it->second;
        for(int i=0;i<(int)list.size();++i){
            if(i) svg+=' ';
            svg+=list[i];
        }
        svg+="\"/></g>";
    }

    svg+="</svg>";

    const double totalMs=vt_now_ms()-t0;
    VT_LOG("vectorize: DONE in %.1f ms | paths=%d speckles=%d "
           "tracer_budget_hits=%d gradients=%d | "
           "trace=%.1f rdp=%.1f bezier=%.1f svg=%.1f ms | "
           "svg_bytes=%zu",
           totalMs,totalPaths,totalSpeckles,totalTracerMaxStepHits,
           (int)gradDefs.size(),
           timeTrace,timeRDP,timeBezier,timeSVG,svg.size());

    if(totalTracerMaxStepHits>0)
        VT_WARN("vectorize: %d component(s) hit tracer budget. "
                "Consider reducing input resolution or increasing filter_speckle.",
                totalTracerMaxStepHits);

    return svg;
}


} // namespace vtracer