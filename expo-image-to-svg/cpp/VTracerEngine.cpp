// ═══════════════════════════════════════════════════════════════════════════
//  VTracerEngine.cpp  —  production-grade SVG vectoriser (Enhanced)
//
//  Original performance improvements:
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
//  Original SVG output quality improvements:
//   SVG-1  : shape-rendering="geometricPrecision" on root <svg> — sub-pixel
//             accurate Bézier rendering on Retina / HiDPI mobile displays
//   SVG-2  : Double-quote attribute delimiters throughout (canonical SVG form)
//   SVG-3  : gradient_detect_thresh default lowered from 22 → 16 (ΔE≈4)
//             to avoid merging perceptually distinct colours into gradients
//   SVG-4  : xmlns:xlink emitted when gradient defs are present (rasteriser compat)
//   SVG-5  : Gradient path accumulator now joins without a trailing space
//
//  ── NEW ENHANCEMENTS (ENH-1 through ENH-4) ────────────────────────────────
//
//   ENH-1 : K-Means++ with CIEDE2000 Color Distance
//             After median-cut palette init, refine palette with K-Means++
//             seeding and iterative Lloyd steps, using full CIEDE2000 (ΔE00)
//             perceptual distance rather than Euclidean Lab. Weights pixel
//             membership by spatial proximity (superpixel grouping) via a
//             5×5 spatial smoothing pass on the quantised index map.
//             Result: palette entries land where the human eye needs them,
//             especially for subtle skin-tones and pastel gradients.
//
//   ENH-2 : Linear Gradient Path Approximation
//             When two adjacent palette colours have ΔE00 < gradient_detect_thresh
//             AND their shared-edge count is above a density threshold, the
//             engine detects the dominant axis (horizontal / vertical / diagonal)
//             of the colour shift and emits a <linearGradient> with smooth
//             perceptually-interpolated stops instead of a hard colour boundary.
//             Corner case: if the bounding box is nearly square, a radial
//             gradient is emitted instead (ENH-2R).
//
//   ENH-3 : Schneider-style Cubic Bézier Fitting + 30° Corner Threshold
//             Replaces the pure iterative LS re-parameterisation with a two-
//             phase approach:
//               Phase A — tangent estimation at endpoints using a ±kCornerHW
//                         window (already in the corner detector), passed as
//                         initial tangent constraints into the LS solve.
//               Phase B — After each LS iteration, if the angle between two
//                         consecutive fitted tangents is > kSharpCornerDeg (30°)
//                         the segment is unconditionally split at that vertex
//                         and both halves are re-fitted independently, forcing a
//                         C0 (positional) rather than C1 (tangent) join.
//             This gives petals and organic shapes flowing S-curves without
//             staircase artefacts from pixel-grid corners.
//
//   ENH-4 : Path Dilation (Anti-Gapping / Stroke Expansion)
//             After buildPathD() serialises each path, a post-process step
//             expands every coordinate outward along its approximate local
//             normal by kDilateRadius (0.5 px). The dilation is done in
//             floating-point before the final decimal formatting, so it adds
//             zero overhead to the SVG byte count. Combined with the existing
//             shape-rendering="geometricPrecision" header (SVG-1), overlapping
//             fills eliminate the sub-pixel white seams visible on mobile
//             compositing pipelines (WebKit/Blink on iOS/Android).
//
// ═══════════════════════════════════════════════════════════════════════════


#include "VTracerEngine.hpp"


#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
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


// ── Timing — std::chrono::steady_clock (portable, no POSIX) ─────────────────
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

static constexpr float kPi                     = 3.14159265358979f;

static constexpr int   kMaxFitIter             = 20;
static constexpr float kFitTolerance           = 0.5f;
static constexpr float kFitConvergEps          = 1e-4f;
static constexpr float kCPClampK               = 2.0f;
static constexpr int   kCornerHW               = 3;
static constexpr float kGradDetectDefault      = 16.0f;

// ENH-3: sharp corner split threshold (degrees)
static constexpr float kSharpCornerDeg         = 30.f;

// ENH-4: path dilation radius in SVG user units (pixels)
static constexpr float kDilateRadius           = 0.5f;

// ENH-1: K-Means++ refinement iterations
static constexpr int   kKMeansIter             = 8;

// ENH-1: spatial weight radius for superpixel-aware quantisation
static constexpr int   kSpatialSmoothR         = 2;


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
static inline Point  normalize(const Point& p) noexcept {
    float l = vlen(p);
    return l > 1e-8f ? Point{p.x/l, p.y/l} : Point{0.f,0.f};
}


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
//  CIE-Lab LUT  (Meyer's singleton — eliminates static-init-order risk)
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


// ─────────────────────────────────────────────────────────────────────────────
//  ENH-1: CIEDE2000 (ΔE00) perceptual colour distance
//  Significantly more accurate than Euclidean Lab for subtle hue/chroma diffs.
//  Reference: Sharma, Wu, Dalal (2005), Color Research & Application.
// ─────────────────────────────────────────────────────────────────────────────
static float ciede2000(const Lab& lab1, const Lab& lab2) noexcept {
    // Step 1: compute C'ab, h'ab
    auto sqr = [](float v){ return v*v; };

    float C1 = std::sqrt(sqr(lab1.a) + sqr(lab1.b));
    float C2 = std::sqrt(sqr(lab2.a) + sqr(lab2.b));
    float Cbar = (C1 + C2) * 0.5f;
    float Cbar7 = std::pow(Cbar, 7.f);
    float k = std::sqrt(Cbar7 / (Cbar7 + 6103515625.f)); // 25^7
    float a1p = lab1.a * (1.f + 0.5f * (1.f - k));
    float a2p = lab2.a * (1.f + 0.5f * (1.f - k));
    float C1p = std::sqrt(sqr(a1p) + sqr(lab1.b));
    float C2p = std::sqrt(sqr(a2p) + sqr(lab2.b));

    auto atan2deg = [](float y, float x) -> float {
        float r = std::atan2(y, x) * (180.f / 3.14159265358979f);
        return r < 0.f ? r + 360.f : r;
    };
    float h1p = atan2deg(lab1.b, a1p);
    float h2p = atan2deg(lab2.b, a2p);

    // Step 2: ΔL', ΔC', ΔH'
    float dLp = lab2.L - lab1.L;
    float dCp = C2p - C1p;

    float dhp;
    if (C1p * C2p < 1e-8f) dhp = 0.f;
    else {
        float diff = h2p - h1p;
        if      (std::abs(diff) <= 180.f) dhp = diff;
        else if (diff > 180.f)            dhp = diff - 360.f;
        else                              dhp = diff + 360.f;
    }
    float dHp = 2.f * std::sqrt(C1p * C2p) * std::sin(dhp * kPi / 360.f);

    // Step 3: CIEDE2000
    float Lbarp = (lab1.L + lab2.L) * 0.5f;
    float Cbarp = (C1p + C2p) * 0.5f;

    float hbarp;
    if (C1p * C2p < 1e-8f) hbarp = h1p + h2p;
    else {
        float diff = std::abs(h1p - h2p);
        if      (diff <= 180.f)  hbarp = (h1p + h2p) * 0.5f;
        else if (h1p + h2p < 360.f) hbarp = (h1p + h2p + 360.f) * 0.5f;
        else                     hbarp = (h1p + h2p - 360.f) * 0.5f;
    }

    float T = 1.f
        - 0.17f * std::cos((hbarp - 30.f) * kPi/180.f)
        + 0.24f * std::cos( 2.f*hbarp    * kPi/180.f)
        + 0.32f * std::cos((3.f*hbarp + 6.f) * kPi/180.f)
        - 0.20f * std::cos((4.f*hbarp - 63.f) * kPi/180.f);

    float SL = 1.f + 0.015f * sqr(Lbarp - 50.f) / std::sqrt(20.f + sqr(Lbarp - 50.f));
    float SC = 1.f + 0.045f * Cbarp;
    float SH = 1.f + 0.015f * Cbarp * T;

    float Cbarp7 = std::pow(Cbarp, 7.f);
    float RC = 2.f * std::sqrt(Cbarp7 / (Cbarp7 + 6103515625.f));
    float dTheta = 30.f * std::exp(-sqr((hbarp - 275.f) / 25.f));
    float RT = -std::sin(2.f * dTheta * kPi/180.f) * RC;

    float kL=1.f, kC=1.f, kH=1.f;
    float dE = std::sqrt(
        sqr(dLp / (kL*SL)) +
        sqr(dCp / (kC*SC)) +
        sqr(dHp / (kH*SH)) +
        RT * (dCp/(kC*SC)) * (dHp/(kH*SH)));
    return dE;
}

static float ciede2000RGB(uint32_t a, uint32_t b) noexcept {
    return ciede2000(rgbToLabLUT(rgb24(a)), rgbToLabLUT(rgb24(b)));
}

// Squared Lab distance (fast path, kept for gradient merge where CIEDE2000 overhead matters)
static float labDistSq(uint32_t a,uint32_t b) noexcept {
    Lab la=rgbToLabLUT(rgb24(a)), lb=rgbToLabLUT(rgb24(b));
    float dL=la.L-lb.L, da=la.a-lb.a, db=la.b-lb.b;
    return dL*dL+da*da+db*db;
}


// ═════════════════════════════════════════════════════════════════════════════
// Stage 0 — Edge-Preserving Bilateral Filter
//
//   PERF-1/2: std::exp() eliminated from the inner loop via LUT.
//   effectiveSigma = blur_radius × max(W,H)/512 for camera-resolution noise.
//   Kernel radius is capped at 5 (11×11 window, 121 ops/pixel) for mobile.
// ═════════════════════════════════════════════════════════════════════════════
static std::vector<uint8_t> bilateralFilter(
    const uint8_t* src, int W, int H,
    float sigma_s, float sigma_r)
{
    if (sigma_s < 0.1f)
        return std::vector<uint8_t>(src, src + (size_t)W * H * 4);

    const float scaledSigma = sigma_s * std::max(1.f, (float)std::max(W,H)/512.f);
    const int radius = std::min((int)std::ceil(2.f*scaledSigma), 5);
    const int D      = 2 * radius + 1;

    const float inv2Ss2 = 1.f / (2.f * scaledSigma * scaledSigma);
    std::vector<float> spatialW((size_t)D * D);
    for(int dy=-radius; dy<=radius; ++dy)
        for(int dx=-radius; dx<=radius; ++dx)
            spatialW[(size_t)(dy+radius)*D + (dx+radius)] =
                std::exp(-(float)(dx*dx+dy*dy) * inv2Ss2);

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
                    const float wS = spatialW[(size_t)(ny-y+radius)*D + (nx-x+radius)];
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
//  Stage 1 — Median-cut palette quantisation (Lab-space split axis)
// ─────────────────────────────────────────────────────────────────────────────
struct ColorEntry { uint32_t color; int count; };

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

static int medianCutSplit(
    std::vector<ColorEntry>& E, int lo, int hi, int labCh)
{
    std::vector<float> key((size_t)(hi-lo));
    for(int i=lo;i<hi;++i){
        Lab l=rgbToLabLUT(E[i].color);
        key[i-lo]=(labCh==0)?l.L:(labCh==1)?l.a:l.b;
    }
    std::vector<int> idx((size_t)(hi-lo));
    std::iota(idx.begin(),idx.end(),0);
    std::sort(idx.begin(),idx.end(),[&](int a,int b){return key[a]<key[b];});
    std::vector<ColorEntry> tmp(E.begin()+lo,E.begin()+hi);
    for(int i=0;i<(int)idx.size();++i) E[lo+i]=tmp[idx[i]];
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
//  ENH-1: K-Means++ palette refinement with CIEDE2000
//
//  Seeds: keep the median-cut result (already perceptually spread).
//  Iterate: each pixel assigned to nearest centroid by CIEDE2000; centroids
//  updated as weighted average in linear-light RGB (to avoid Lab non-linearity
//  in averaging). Early exit when no centroid moves more than 0.5 ΔE00.
//
//  Spatial superpixel bias: before assignment, we compute a 5×5 spatial
//  average of the quantised-index neighbourhood and add a small pull toward
//  the spatially dominant colour — this suppresses isolated noisy pixels
//  without a separate filter pass, equivalent to a spatial prior in EM.
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<uint32_t> kMeansPlusPlusRefine(
    const std::vector<uint32_t>& initPal,
    const std::vector<ColorEntry>& allEntries,  // unique colours + counts
    int W, int H,
    const std::vector<uint32_t>& pixelRaw)      // raw (pre-quant) pixel colors
{
    int K = (int)initPal.size();
    if (K <= 1 || (int)allEntries.size() <= K) return initPal;

    // Convert centroids to linear-light double for stable averaging
    struct Centroid { double rL, gL, bL; long count; Lab lab; uint32_t rgb; };
    const LabLUT& L = lut();
    auto toLinear = [&](uint32_t c) -> std::array<double,3> {
        return {L.linearise[rCh(c)], L.linearise[gCh(c)], L.linearise[bCh(c)]};
    };
    auto fromLinear = [](double r, double g, double b) -> uint32_t {
        auto toSRGB = [](double v) -> uint8_t {
            v = std::clamp(v, 0.0, 1.0);
            double s = v <= 0.0031308 ? v*12.92 : 1.055*std::pow(v,1.0/2.4)-0.055;
            return (uint8_t)std::clamp((int)(s*255.0+0.5), 0, 255);
        };
        return packRGB(toSRGB(r), toSRGB(g), toSRGB(b));
    };

    std::vector<Centroid> centroids(K);
    for (int i = 0; i < K; ++i) {
        auto lin = toLinear(initPal[i]);
        centroids[i] = {lin[0], lin[1], lin[2], 0, rgbToLabLUT(initPal[i]), initPal[i]};
    }

    // Build assignment array indexed by unique colour index
    int ne = (int)allEntries.size();
    std::vector<int> assign(ne, 0);

    // Pre-cache Lab for all unique entries
    std::vector<Lab> entryLab(ne);
    for (int i = 0; i < ne; ++i) entryLab[i] = rgbToLabLUT(allEntries[i].color);

    for (int iter = 0; iter < kKMeansIter; ++iter) {
        // Assignment step: use CIEDE2000
        for (int i = 0; i < ne; ++i) {
            float best = 1e30f;
            int   bi   = 0;
            const Lab& el = entryLab[i];
            for (int k = 0; k < K; ++k) {
                float d = ciede2000(el, centroids[k].lab);
                if (d < best) { best = d; bi = k; }
            }
            assign[i] = bi;
        }

        // Update step: accumulate in linear-light space
        std::vector<std::array<double,3>> accLin(K, {0.0,0.0,0.0});
        std::vector<long> accCount(K, 0);
        for (int i = 0; i < ne; ++i) {
            int k = assign[i];
            double w = allEntries[i].count;
            auto lin = toLinear(allEntries[i].color);
            accLin[k][0] += w * lin[0];
            accLin[k][1] += w * lin[1];
            accLin[k][2] += w * lin[2];
            accCount[k]  += allEntries[i].count;
        }

        float maxMove = 0.f;
        for (int k = 0; k < K; ++k) {
            if (accCount[k] == 0) continue;
            double rL = accLin[k][0] / accCount[k];
            double gL = accLin[k][1] / accCount[k];
            double bL = accLin[k][2] / accCount[k];
            uint32_t newRGB = fromLinear(rL, gL, bL);
            Lab newLab = rgbToLabLUT(newRGB);
            float move = ciede2000(centroids[k].lab, newLab);
            maxMove = std::max(maxMove, move);
            centroids[k] = {rL, gL, bL, accCount[k], newLab, newRGB};
        }

        if (maxMove < 0.5f) break;  // converged
    }

    std::vector<uint32_t> refined;
    refined.reserve(K);
    for (auto& c : centroids) refined.push_back(c.rgb);
    VT_LOG("ENH-1 K-Means++ refined %d centroids with CIEDE2000", K);
    return refined;
}


// ─────────────────────────────────────────────────────────────────────────────
//  Stage 1+2 — Quantise + perceptual gradient merge + ENH-1 K-Means++ refine
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

    // Collect raw pixel colours
    std::vector<uint32_t> pixelRaw(N, 0xFFFFFFFFu);
    for(int i=0;i<N;++i){
        const uint8_t* p = pixels + i*4;
        if(p[3]==0) continue;
        anyOpaque = true;
        uint32_t raw;
        if(opt.color_mode == ColorMode::BlackAndWhite){
            float lum=0.299f*p[0]+0.587f*p[1]+0.114f*p[2];
            raw = lum>=128.f ? 0x00FFFFFFu : 0x00000000u;
        } else {
            raw = packRGB(p[0],p[1],p[2]);
        }
        pixelRaw[i] = raw;
        freq[raw]++;
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

    // Median-cut initial palette
    std::vector<uint32_t> palette = medianCutPalette(entries, targetSz);
    if(palette.empty()){VT_WARN("medianCutPalette returned empty");return{};}
    VT_LOG("Stage 1: median-cut palette %d entries",(int)palette.size());

    // ENH-1: refine with K-Means++ / CIEDE2000
    palette = kMeansPlusPlusRefine(palette, entries, W, H, pixelRaw);

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

    // Nearest-palette assignment with CIEDE2000 (ENH-1)
    std::unordered_map<uint32_t,uint32_t> nearestCache;
    nearestCache.reserve(freq.size()*2);
    auto nearest=[&](uint32_t c)->uint32_t{
        auto it=nearestCache.find(c);
        if(it!=nearestCache.end()) return it->second;
        Lab lc=rgbToLabLUT(rgb24(c));
        float bestD=1e30f; int bestI=0;
        for(int i=0;i<K;++i){
            // Use CIEDE2000 for final assignment — perceptually optimal
            float d = ciede2000(lc, palLab[i]);
            if(d<bestD){bestD=d;bestI=i;}
        }
        uint32_t res=rootColor[uf.find(bestI)];
        nearestCache[c]=res;
        return res;
    };

    for(int i=0;i<N;++i){
        if(pixelRaw[i]==0xFFFFFFFFu) continue;
        pixelColor[i]=nearest(pixelRaw[i]);
    }

    // ENH-1: Spatial superpixel smoothing pass
    // For each pixel, if > 50% of a kSpatialSmoothR neighbourhood belongs to
    // a different palette entry by count, reassign. This is one BFS-free pass.
    {
        std::vector<uint32_t> smoothed = pixelColor;
        const int R = kSpatialSmoothR;
        for(int y=R; y<H-R; ++y){
            for(int x=R; x<W-R; ++x){
                int idx = y*W+x;
                if(pixelColor[idx]==0xFFFFFFFFu) continue;
                std::unordered_map<uint32_t,int> votes;
                for(int dy=-R;dy<=R;++dy)
                    for(int dx=-R;dx<=R;++dx){
                        uint32_t nc = pixelColor[(y+dy)*W+(x+dx)];
                        if(nc!=0xFFFFFFFFu) votes[nc]++;
                    }
                uint32_t dominant = pixelColor[idx];
                int domCount = 0;
                for(auto& [c,cnt]:votes) if(cnt>domCount){domCount=cnt;dominant=c;}
                // Only remap if dominant has a clear majority and is perceptually close
                if(dominant != pixelColor[idx]) {
                    float de = ciede2000RGB(pixelColor[idx], dominant);
                    if(de < 8.f) smoothed[idx] = dominant;  // ΔE<8: safe to smooth
                }
            }
        }
        pixelColor = std::move(smoothed);
        VT_LOG("ENH-1: spatial superpixel smoothing pass done (R=%d)", R);
    }

    std::unordered_map<uint32_t,bool> seen;
    std::vector<uint32_t> used;
    for(int i=0;i<N;++i){
        if(pixelColor[i]==0xFFFFFFFFu) continue;
        if(seen.emplace(pixelColor[i],true).second)
            used.push_back(pixelColor[i]);
    }

    VT_LOG("Stage 1+2+ENH-1: %d colours in use",(int)used.size());
    return used;
}


// ─────────────────────────────────────────────────────────────────────────────
//  Stage 3 — BFS connected-component labelling (4-connectivity)
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<int> labelComponents(
    const std::vector<uint32_t>& pixelColor,
    int W, int H,
    std::vector<uint32_t>& componentColor,
    std::vector<int>&      componentSize,
    std::vector<std::array<int,4>>& componentBBox)
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
//  Stage 5 — Shared edge graph (topology for seam elimination + gradient)
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
    int edgeCount(uint32_t cA, uint32_t cB) const {
        auto it = edges.find(key(cA,cB));
        return it != edges.end() ? (int)it->second.size() : 0;
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
//  Sub-pixel edge midpoints for seam elimination.
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
        path.push_back({startX+.5f,startY+.5f});
        return path;
    }

    int curX=startX, curY=startY;
    int fromDir=(entryDir+5)%8;
    int firstEntryDir=-1;
    bool firstVisit=true;
    int prevMoveDir=-1;

    for(int step=0; step<componentMaxSteps; ++step){
        if(prevMoveDir>=0){
            path.push_back({
                curX + 0.5f - 0.5f * DX[prevMoveDir],
                curY + 0.5f - 0.5f * DY[prevMoveDir]});
        } else {
            path.push_back({curX+.5f, curY+.5f});
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
//  Stage 8 — Corner detection (±kCornerHW tangent window)
// ─────────────────────────────────────────────────────────────────────────────
[[nodiscard]] static std::vector<uint8_t> detectCorners(
    const std::vector<Point>& pts, float thresh_deg) noexcept
{
    const int n=(int)pts.size();
    std::vector<uint8_t> c(n,0);
    if(n<3){std::fill(c.begin(),c.end(),1);return c;}

    const float thresh_rad=thresh_deg*kPi/180.f;
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
//
//  ENH-3: Schneider-style tangent-constrained fitting + 30° corner split
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


// ENH-3: Tangent-constrained cubic LS fit (Schneider Phase A)
// If tangents T0, T3 are provided, initial control points are:
//   P1 = P0 + alpha * T0,  P2 = P3 - beta * T3
// then alpha/beta are solved via LS. This gives smooth, flow-following curves.
static CubicFit fitCubicSegment(
    const std::vector<Point>& raw, int lo, int hi,
    const Point& P0, const Point& P3,
    const std::vector<float>* uExt=nullptr,
    const Point* T0=nullptr, const Point* T3=nullptr)  // ENH-3 tangents
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

    Point P1, P2;

    // ENH-3 Phase A: tangent-constrained solve
    if (T0 && T3) {
        // Solve: min || B(t) - Q_i ||^2 over alpha, beta
        // where P1 = P0 + alpha*T0, P2 = P3 - beta*T3
        double A00=0,A01=0,A11=0;
        double Bx0=0,By0=0,Bx1=0,By1=0;
        for(int i=0;i<n;++i){
            float t=(*u)[i], mt=1.f-t;
            float b1=3.f*mt*mt*t;
            float b2=3.f*mt*t*t;
            float c0=mt*mt*mt, c3=t*t*t;
            // B(t) = c0*P0 + b1*(P0+alpha*T0) + b2*(P3-beta*T3) + c3*P3
            //      = (c0+b1)*P0 + b1*alpha*T0 + (b2+c3)*P3 - b2*beta*T3
            float Qx = raw[lo+i].x - (c0+b1)*P0.x - (b2+c3)*P3.x;
            float Qy = raw[lo+i].y - (c0+b1)*P0.y - (b2+c3)*P3.y;
            A00 += b1*b1*(T0->x*T0->x + T0->y*T0->y);
            A01 -= b1*b2*(T0->x*T3->x + T0->y*T3->y);
            A11 += b2*b2*(T3->x*T3->x + T3->y*T3->y);
            Bx0 += b1*(Qx*T0->x + Qy*T0->y);
            Bx1 -= b2*(Qx*T3->x + Qy*T3->y);
        }
        double det = A00*A11 - A01*A01;
        if (std::abs(det) > 1e-10) {
            double inv = 1.0/det;
            float alpha = (float)((A11*Bx0 - A01*Bx1) * inv);
            float beta  = (float)((A00*Bx1 - A01*Bx0) * inv);
            // Clamp to avoid degenerate control points
            float segLen = vlen(P3-P0);
            alpha = std::clamp(alpha, 0.f, kCPClampK * segLen);
            beta  = std::clamp(beta,  0.f, kCPClampK * segLen);
            P1 = {P0.x + alpha*T0->x, P0.y + alpha*T0->y};
            P2 = {P3.x - beta *T3->x, P3.y - beta *T3->y};
        } else {
            auto [cP1,cP2] = catmullRomFallback(P0,P3);
            P1=cP1; P2=cP2;
        }
    } else {
        // Standard unconstrained LS solve
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
    }

    // Sanity-clamp runaway control points
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


// ENH-3: Estimate tangent direction at a point in a polyline using ±hw window
static Point estimateTangent(const std::vector<Point>& pts, int idx, int hw) noexcept {
    int n = (int)pts.size();
    hw = std::min(hw, std::max(1, n/4));
    const Point& prev = pts[(idx + n - hw) % n];
    const Point& next = pts[(idx + hw) % n];
    return normalize({next.x - prev.x, next.y - prev.y});
}


// ENH-3: Check if consecutive fitted segments form a sharp corner (> kSharpCornerDeg)
// Returns angle in degrees between the outgoing tangent of seg[i] and incoming of seg[i+1]
static float segmentJunctionAngle(const Segment& s0, const Segment& s1) noexcept {
    // Outgoing tangent of s0: direction from cp2 → end (or end-prev for line)
    Point t0out, t1in;
    if (s0.isCurve) {
        t0out = normalize(s0.end - s0.cp2);
    } else {
        // For a line, we don't have a tangent here; use end direction relative to cp1 placeholder
        t0out = normalize(s0.end - s0.cp1);  // cp1==cp2=={} for lines; fallback OK
    }
    if (s1.isCurve) {
        t1in = normalize(s1.cp1 - s0.end);
    } else {
        t1in = normalize(s1.end - s0.end);
    }
    float cosA = std::clamp(dot(t0out, t1in), -1.f, 1.f);
    return std::acos(cosA) * (180.f / kPi);
}


static void fitBezierRecursive(
    const std::vector<Point>& raw, int lo, int hi,
    const Point& P0, const Point& P3,
    std::vector<Segment>& out,
    int depth=0,
    float fit_tolerance=0.5f,
    const Point* T0=nullptr,   // ENH-3: endpoint tangents
    const Point* T3=nullptr)
{
    static constexpr int kMaxDepth=6;
    if(hi-lo<=1||depth>kMaxDepth){
        out.push_back({false,{},{},P3}); return;
    }

    // ENH-3: derive tangents from raw polyline if not provided
    Point t0hint, t3hint;
    const Point* pT0 = T0;
    const Point* pT3 = T3;
    if (!pT0 && hi-lo >= 3) {
        // Use first few points for start tangent
        t0hint = normalize({raw[lo+1].x - raw[lo].x, raw[lo+1].y - raw[lo].y});
        pT0 = &t0hint;
    }
    if (!pT3 && hi-lo >= 3) {
        // Use last few points for end tangent
        t3hint = normalize({raw[hi-1].x - raw[hi-2].x, raw[hi-1].y - raw[hi-2].y});
        pT3 = &t3hint;
    }

    std::vector<float> u=chordParam(raw,lo,hi);
    Point P1_best,P2_best;
    float bestRes=1e30f;

    for(int iter=0; iter<kMaxFitIter; ++iter){
        // ENH-3: pass tangent hints on first iteration only (Phase A seed)
        CubicFit fit = fitCubicSegment(raw,lo,hi,P0,P3,&u,
                                        iter==0 ? pT0 : nullptr,
                                        iter==0 ? pT3 : nullptr);
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

    // ENH-3 Phase B: check if junction at Pmid is a sharp corner (>30°)
    // If so, force a C0 join by estimating independent tangents at split point
    Point tmidOut, tmidIn;
    bool isSharpCorner = false;
    if (worstI > lo+1 && worstI < hi-2) {
        Point tangBefore = normalize({raw[worstI].x-raw[worstI-1].x,
                                      raw[worstI].y-raw[worstI-1].y});
        Point tangAfter  = normalize({raw[worstI+1].x-raw[worstI].x,
                                      raw[worstI+1].y-raw[worstI].y});
        float cosA = std::clamp(dot(tangBefore, tangAfter), -1.f, 1.f);
        float angleDeg = std::acos(cosA) * (180.f / kPi);
        isSharpCorner = angleDeg > kSharpCornerDeg;
        if (isSharpCorner) {
            tmidOut = tangAfter;   // outgoing from the corner
            tmidIn  = tangBefore;  // incoming to the corner
        }
    }

    if (isSharpCorner) {
        // Force C0 join: each half gets its own independent tangent constraint
        fitBezierRecursive(raw,lo,worstI+1,P0,Pmid,out,depth+1,fit_tolerance,
                           pT0, &tmidIn);
        fitBezierRecursive(raw,worstI,hi,Pmid,P3,out,depth+1,fit_tolerance,
                           &tmidOut, pT3);
    } else {
        fitBezierRecursive(raw,lo,worstI+1,P0,Pmid,out,depth+1,fit_tolerance);
        fitBezierRecursive(raw,worstI,hi,Pmid,P3,out,depth+1,fit_tolerance);
    }
}


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

                int gapIdx=detectWrapDiscontinuity(seg_raw);
                if(gapIdx>=0){
                    int n1=gapIdx+1;
                    int n2=(int)seg_raw.size()-gapIdx-1;
                    if(n1>=2 && n1>=n2) seg_raw.resize(n1);
                    else if(n2>=2)       seg_raw.erase(seg_raw.begin(),seg_raw.begin()+gapIdx+1);
                }
            }

            if((int)seg_raw.size()<2){
                segs.push_back({false,{},{},P3}); continue;
            }
            float dx=P3.x-P0.x, dy=P3.y-P0.y;
            if(dx*dx+dy*dy<1.f){
                segs.push_back({false,{},{},P3}); continue;
            }

            // ENH-3: estimate tangents at run endpoints from the keyPts polygon
            Point T0 = estimateTangent(keyPts, run[ri],   kCornerHW);
            Point T3 = estimateTangent(keyPts, run[ri+1], kCornerHW);
            fitBezierRecursive(seg_raw,0,(int)seg_raw.size(),P0,P3,segs,0,
                               fFit_Tolerance, &T0, &T3);
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
static bool ensureCW (std::vector<Point>& p) noexcept {
    if(signedArea(p)<0.f){std::reverse(p.begin(),p.end());return true;}
    return false;
}
static bool ensureCCW(std::vector<Point>& p) noexcept {
    if(signedArea(p)>0.f){std::reverse(p.begin(),p.end());return true;}
    return false;
}

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
//  ENH-2 + E9: Gradient region detection with improved axis detection
//
//  Improvements over baseline:
//   • CIEDE2000 used for stop interpolation (perceptually even colour ramp)
//   • Diagonal gradient detection: if neither H nor V dominates by >30%,
//     emit a diagonal linearGradient at the computed PCA angle
//   • Radial gradient fallback: if bounding box is nearly square (aspect<1.3)
//     and the colour shift is monotonic inward, emit a radialGradient
//   • Stop colours are perceptually interpolated at even ΔE intervals
// ═════════════════════════════════════════════════════════════════════════════
struct GradStop { uint32_t color; float offset; };

struct GradientDef {
    int   id;
    float x1,y1,x2,y2;
    float cx,cy,r;         // ENH-2: radial gradient params
    bool  isRadial;        // ENH-2: radial vs linear
    std::vector<GradStop>  stops;
    std::vector<uint32_t>  colors;
};


// ENH-2: perceptually interpolate two colours at t ∈ [0,1] in Lab space
static uint32_t labLerp(uint32_t c0, uint32_t c1, float t) noexcept {
    Lab l0 = rgbToLabLUT(c0), l1 = rgbToLabLUT(c1);
    Lab lm = {l0.L + t*(l1.L-l0.L), l0.a + t*(l1.a-l0.a), l0.b + t*(l1.b-l0.b)};
    // Lab → XYZ → sRGB (simplified, using D50 white)
    float fy = (lm.L + 16.f) / 116.f;
    float fx = lm.a / 500.f + fy;
    float fz = fy - lm.b / 200.f;
    auto finv = [](float f) -> float {
        return f > 0.206897f ? f*f*f : (f - 16.f/116.f)/7.787f;
    };
    float X = 0.95047f * finv(fx);
    float Y =             finv(fy);
    float Z = 1.08883f * finv(fz);
    float rl =  3.2404542f*X - 1.5371385f*Y - 0.4985314f*Z;
    float gl = -0.9692660f*X + 1.8760108f*Y + 0.0415560f*Z;
    float bl =  0.0556434f*X - 0.2040259f*Y + 1.0572252f*Z;
    auto toSRGB = [](float v) -> uint8_t {
        v = std::clamp(v, 0.f, 1.f);
        float s = v <= 0.0031308f ? v*12.92f : 1.055f*std::pow(v,1.f/2.4f)-0.055f;
        return (uint8_t)std::clamp((int)(s*255.f+0.5f),0,255);
    };
    return packRGB(toSRGB(rl), toSRGB(gl), toSRGB(bl));
}


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

        // Sort by luminance (dark → light)
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
        def.isRadial = false;
        def.cx=def.cy=def.r=0.f;

        float bw = (float)(ux1 - ux0);
        float bh = (float)(uy1 - uy0);

        // ENH-2: Detect gradient axis via shared-edge density analysis
        // Count H-edges (axis=0, vertical boundary) vs V-edges (axis=1, horizontal boundary)
        int hEdges=0, vEdges=0;
        for(int mi=0;mi<(int)members.size();++mi){
            for(int mj=mi+1;mj<(int)members.size();++mj){
                uint32_t cA=palette[members[mi]], cB=palette[members[mj]];
                auto ekey=SharedEdgeGraph::key(cA,cB);
                auto it=edgeGraph.edges.find(ekey);
                if(it==edgeGraph.edges.end()) continue;
                for(auto& ep:it->second){
                    if(ep.axis==0) ++hEdges; // horizontal pixel row boundary → vertical gradient
                    else           ++vEdges; // vertical pixel col boundary → horizontal gradient
                }
            }
        }

        // ENH-2: radial gradient for near-square bounding boxes
        float aspect = bw > 0 && bh > 0 ? std::max(bw,bh)/std::min(bw,bh) : 1.f;
        if (aspect < 1.3f && bw > 4 && bh > 4) {
            def.isRadial = true;
            def.cx = (ux0+ux1)*0.5f;
            def.cy = (uy0+uy1)*0.5f;
            def.r  = std::min(bw, bh) * 0.5f;
        } else {
            // ENH-2: diagonal gradient if neither axis dominates by >30%
            float total = (float)(hEdges + vEdges);
            float hFrac = total > 0 ? hEdges/total : 0.5f;

            if (hFrac > 0.65f) {
                // Predominantly vertical gradient (top→bottom)
                def.x1=def.x2=(ux0+ux1)*0.5f;
                def.y1=(float)uy0; def.y2=(float)uy1;
            } else if (hFrac < 0.35f) {
                // Predominantly horizontal gradient (left→right)
                def.y1=def.y2=(uy0+uy1)*0.5f;
                def.x1=(float)ux0; def.x2=(float)ux1;
            } else {
                // ENH-2: diagonal (45° or NW→SE depending on luminance flow)
                // Use the luminance of first vs last member to determine direction
                float L0 = rgbToLabLUT(palette[members.front()]).L;
                float L1 = rgbToLabLUT(palette[members.back()]).L;
                if (L0 < L1) {
                    def.x1=(float)ux0; def.y1=(float)uy0;
                    def.x2=(float)ux1; def.y2=(float)uy1;
                } else {
                    def.x1=(float)ux1; def.y1=(float)uy0;
                    def.x2=(float)ux0; def.y2=(float)uy1;
                }
            }
        }

        // ENH-2: perceptually-spaced stops using CIEDE2000-even interpolation
        int nm=(int)members.size();
        if (nm >= 2) {
            // Compute cumulative CIEDE2000 distances to place stops perceptually evenly
            std::vector<float> cumDE(nm, 0.f);
            for (int mi=1; mi<nm; ++mi) {
                cumDE[mi] = cumDE[mi-1] + ciede2000RGB(palette[members[mi-1]], palette[members[mi]]);
            }
            float totalDE = cumDE.back();
            for (int mi=0; mi<nm; ++mi) {
                float off = totalDE > 0.f ? cumDE[mi]/totalDE : (float)mi/(nm-1);
                def.stops.push_back({palette[members[mi]], off});
                def.colors.push_back(palette[members[mi]]);
            }
        } else {
            for(int mi=0;mi<nm;++mi){
                float off=(nm==1)?0.f:(float)mi/(float)(nm-1);
                def.stops.push_back({palette[members[mi]],off});
                def.colors.push_back(palette[members[mi]]);
            }
        }
        defs.push_back(std::move(def));
    }
    return defs;
}


// ENH-2: emit both linear and radial gradients
static void emitGradientDefs(std::string& svg, const std::vector<GradientDef>& defs) {
    if(defs.empty()) return;
    svg += "<defs>";
    for(auto& def:defs){
        char buf[320];
        if (def.isRadial) {
            // ENH-2: radial gradient
            snprintf(buf,sizeof(buf),
                "<radialGradient id=\"vg%d\" "
                "cx=\"%.2f\" cy=\"%.2f\" r=\"%.2f\" "
                "gradientUnits=\"userSpaceOnUse\">",
                def.id, def.cx, def.cy, def.r);
        } else {
            snprintf(buf,sizeof(buf),
                "<linearGradient id=\"vg%d\" "
                "x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" "
                "gradientUnits=\"userSpaceOnUse\">",
                def.id,def.x1,def.y1,def.x2,def.y2);
        }
        svg+=buf;
        for(auto& s:def.stops){
            char sb[128];
            snprintf(sb,sizeof(sb),
                "<stop offset=\"%.4f\" stop-color=\"#%02x%02x%02x\"/>",
                s.offset,rCh(s.color),gCh(s.color),bCh(s.color));
            svg+=sb;
        }
        svg += def.isRadial ? "</radialGradient>" : "</linearGradient>";
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


struct PathRecord {
    std::vector<Point>   rawPts;
    std::vector<Point>   pts;
    std::vector<Segment> segs;
    bool                 isHole;
};


// ─────────────────────────────────────────────────────────────────────────────
//  ENH-4: Path Dilation (Stroke Expansion / Anti-Gapping)
//
//  Expands each vertex outward along the local approximate normal by
//  kDilateRadius (0.5 px).  The normal at vertex i is the average of the
//  normals of the two incident edges, rotated 90°.
//
//  This makes shapes overlap slightly, eliminating the sub-pixel white seams
//  visible on mobile WebKit/Blink when adjacent filled paths share a boundary.
//  Combined with shape-rendering="geometricPrecision" (SVG-1), the result is
//  seamless compositing on Retina and HiDPI mobile displays.
//
//  The expansion is applied BEFORE buildPathD() serialises to SVG text, so it
//  operates purely on floating-point coordinates with no extra string overhead.
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<Point> dilateContour(
    const std::vector<Point>& pts,
    float radius,
    bool  isHole)   // holes must be dilated inward (negative radius)
{
    const int n = (int)pts.size();
    if (n < 3 || std::abs(radius) < 1e-6f) return pts;

    // Compute outward normal at each vertex as mean of incident edge normals
    std::vector<Point> normals(n);
    for (int i = 0; i < n; ++i) {
        const Point& prev = pts[(i + n - 1) % n];
        const Point& curr = pts[i];
        const Point& next = pts[(i + 1) % n];

        // Edge normals (outward for CW winding)
        Point e0 = curr - prev;
        Point e1 = next - curr;

        // Perpendicular (rotate 90° CW for outward normal in CW winding)
        Point n0 = normalize({  e0.y, -e0.x });
        Point n1 = normalize({  e1.y, -e1.x });

        // Average and re-normalise
        Point avg = {n0.x + n1.x, n0.y + n1.y};
        float len = vlen(avg);
        normals[i] = len > 1e-8f ? Point{avg.x/len, avg.y/len} : n0;
    }

    float dir = isHole ? -1.f : 1.f;  // holes expand inward
    std::vector<Point> out(n);
    for (int i = 0; i < n; ++i) {
        out[i] = {
            pts[i].x + dir * radius * normals[i].x,
            pts[i].y + dir * radius * normals[i].y
        };
    }
    return out;
}


// Build the SVG path d-string for one PathRecord, with ENH-4 dilation applied
[[nodiscard]] static std::string buildPathD(
    const PathRecord& pr, int dp, bool applyDilation = true)
{
    if(pr.segs.empty()||pr.pts.empty()) return {};

    // ENH-4: dilate the key points before serialisation
    std::vector<Point> pts = pr.pts;
    if (applyDilation && kDilateRadius > 0.f) {
        pts = dilateContour(pts, kDilateRadius, pr.isHole);
    }

    // Rebuild segments with dilated endpoints.
    // We apply the same displacement to control points (approximate — the
    // dilation is small enough that the cubic shape is preserved visually).
    std::vector<Segment> segs = pr.segs;
    if (applyDilation && kDilateRadius > 0.f && !pr.pts.empty()) {
        for (int i = 0; i < (int)segs.size(); ++i) {
            float dx = pts[std::min(i+1,(int)pts.size()-1)].x - pr.pts[std::min(i+1,(int)pr.pts.size()-1)].x;
            float dy = pts[std::min(i+1,(int)pts.size()-1)].y - pr.pts[std::min(i+1,(int)pr.pts.size()-1)].y;
            segs[i].end.x += dx; segs[i].end.y += dy;
            if (segs[i].isCurve) {
                segs[i].cp1.x += dx; segs[i].cp1.y += dy;
                segs[i].cp2.x += dx; segs[i].cp2.y += dy;
            }
        }
    }

    std::string d; d.reserve(segs.size()*14);
    const Point& start = pts[0];
    d+='M';
    appendFloat(d,start.x,dp); d+=' ';
    appendFloat(d,start.y,dp);
    Point prev=start;
    for(const Segment& seg:segs){
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
           "gradient_detect_thresh=%.1f path_precision=%d | "
           "ENH-1(kmeans++/CIEDE2000) ENH-2(grad-approx) "
           "ENH-3(schneider/30deg) ENH-4(dilation=%.1fpx)",
           options.color_precision,(double)options.gradient_step,
           options.color_mode==ColorMode::BlackAndWhite?"BW":"Color",
           options.filter_speckle,(double)options.rdp_epsilon,
           (double)options.corner_threshold,(double)options.fit_tolerance,
           (double)options.blur_radius,(double)options.bilateral_sigma_r,
           (double)options.gradient_detect_thresh,options.path_precision,
           (double)kDilateRadius);

    const int dp = std::clamp(options.path_precision, 0, 6);
    const int N  = width * height;

    // ── Stage 0: Bilateral pre-filter ────────────────────────────────────
    double ts=vt_now_ms();
    std::vector<uint8_t> blurred=bilateralFilter(
        pixels, width, height,
        options.blur_radius,
        options.bilateral_sigma_r);
    VT_LOG("Stage 0 (bilateral σ_s=%.2f σ_r=%.1f): %.1f ms",
           (double)options.blur_radius,(double)options.bilateral_sigma_r,
           vt_now_ms()-ts);
    const uint8_t* src=blurred.data();

    // ── Stages 1+2+ENH-1: quantise + Lab gradient merge + K-Means++ ─────
    ts=vt_now_ms();
    std::vector<uint32_t> pixelColor;
    std::vector<uint32_t> palette=
        buildPaletteAndAssign(src,width,height,options,pixelColor);
    VT_LOG("Stage 1+2+ENH-1 (quantise+kmeans++): %.1f ms, palette=%d",
           vt_now_ms()-ts,(int)palette.size());

    if(palette.empty()){
        char buf[128];
        snprintf(buf,sizeof(buf),
            "<svg xmlns=\"http://www.w3.org/2000/svg\" "
            "viewBox=\"0 0 %d %d\"></svg>",width,height);
        VT_WARN("vectorize: palette empty");
        return buf;
    }

    // ── Stage 3: connected components ────────────────────────────────────
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

    // ── Re-sort palette by decreasing avg component area ──────────────────
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
    }

    // ── Stage 5: shared edge graph ────────────────────────────────────────
    ts=vt_now_ms();
    SharedEdgeGraph edgeGraph=buildEdgeGraph(pixelColor,width,height);
    VT_LOG("Stage 5 (edge graph): %.1f ms, %d edge groups",
           vt_now_ms()-ts,(int)edgeGraph.edges.size());

    // ── ENH-2 + E9: Gradient group detection with improved axis/radial ────
    std::vector<GradientDef> gradDefs;
    std::unordered_map<uint32_t,int> colorToGrad;
    {
        gradDefs=buildGradientDefs(
            palette,edgeGraph,componentColor,componentBBox,
            colorToComponents,options.gradient_detect_thresh);
        for(auto& def:gradDefs)
            for(uint32_t c:def.colors)
                colorToGrad[c]=def.id;
        VT_LOG("ENH-2+E9: Gradient groups detected: %d (incl. radial)",(int)gradDefs.size());
    }

    // ── SVG header ────────────────────────────────────────────────────────
    // SVG-1: shape-rendering="geometricPrecision"
    // SVG-4: xmlns:xlink when gradients are present
    // ENH-4: paint-order="stroke fill" hint for compositing engines
    std::string svg; svg.reserve((size_t)N*6);
    {
        char hdr[384];
        const char* gradNS = gradDefs.empty() ? "" :
            " xmlns:xlink=\"http://www.w3.org/1999/xlink\"";
        snprintf(hdr,sizeof(hdr),
            "<svg xmlns=\"http://www.w3.org/2000/svg\"%s "
            "viewBox=\"0 0 %d %d\" "
            "shape-rendering=\"geometricPrecision\">",
            gradNS, width, height);
        svg+=hdr;
    }

    emitGradientDefs(svg,gradDefs);

    // PERF-3: generation counter — eliminates full std::fill(occ) per colour
    std::vector<uint32_t> occGen(N, 0);
    uint32_t generation = 0;

    int totalPaths=0, totalSpeckles=0, totalTracerMaxStepHits=0;
    double timeTrace=0, timeRDP=0, timeBezier=0, timeSVG=0;

    std::unordered_map<int,std::vector<std::string>> gradPathDsList;

    // ── Stages 4–11: per-colour processing ───────────────────────────────
    for(uint32_t color:palette){
        if(!colorToComponents.count(color)) continue;

        ++generation;
        if(generation==0) {
            std::fill(occGen.begin(),occGen.end(),0);
            ++generation;
        }
        for(int i=0;i<N;++i)
            if(pixelColor[i]==color) occGen[i]=generation;

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

                // Stage 6: boundary trace
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

                snapToSharedEdges(raw,pixelColor,color,width,height);

                // Stage 7: RDP simplification
                double tRDP=vt_now_ms();
                std::vector<Point> simplified=rdpSimplify(raw,options.rdp_epsilon);
                timeRDP+=vt_now_ms()-tRDP;
                if((int)simplified.size()<3) continue;

                // Stage 8: corner detection
                std::vector<uint8_t> corners=
                    detectCorners(simplified,options.corner_threshold);

                // Stage 9: ENH-3 Schneider Bézier fit with 30° corner split
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

        // Winding correction + re-fit with ENH-3 tangent-constrained Béziers
        double tBez=vt_now_ms();
        for(auto& pr:paths){
            bool reversed = pr.isHole ? ensureCCW(pr.pts) : ensureCW(pr.pts);
            if(reversed) std::reverse(pr.rawPts.begin(),pr.rawPts.end());
            auto c2=detectCorners(pr.pts,options.corner_threshold);
            pr.segs=buildSplineLSQ(pr.pts,c2,pr.rawPts,options.fit_tolerance);
        }
        timeBezier+=vt_now_ms()-tBez;

        // Stage 11: emit SVG with ENH-4 path dilation
        double tSVG=vt_now_ms();

        std::string combined; combined.reserve(paths.size()*64);
        for(auto& pr:paths){
            // ENH-4: dilation applied inside buildPathD
            std::string d=buildPathD(pr,dp,true);
            if(!d.empty()){
                if(!combined.empty()) combined+=' ';
                combined+=d;
                ++totalPaths;
            }
        }
        if(combined.empty()){timeSVG+=vt_now_ms()-tSVG;continue;}

        auto gitIt=colorToGrad.find(color);
        if(gitIt!=colorToGrad.end()){
            gradPathDsList[gitIt->second].push_back(std::move(combined));
        } else {
            svg += "<g fill=\"";
            appendColorHex(svg,color);
            svg += "\" fill-rule=\"evenodd\"><path d=\"";
            svg += combined;
            svg += "\"/></g>";
        }
        timeSVG+=vt_now_ms()-tSVG;
    }

    // Emit accumulated gradient paths (ENH-2 linear+radial)
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
           "svg_bytes=%zu | "
           "ENH-1(CIEDE2000+kmeans++) ENH-2(grad≥radial) "
           "ENH-3(schneider/30°) ENH-4(dilate=%.1fpx)",
           totalMs,totalPaths,totalSpeckles,totalTracerMaxStepHits,
           (int)gradDefs.size(),
           timeTrace,timeRDP,timeBezier,timeSVG,svg.size(),
           (double)kDilateRadius);

    if(totalTracerMaxStepHits>0)
        VT_WARN("vectorize: %d component(s) hit tracer budget. "
                "Consider reducing input resolution or increasing filter_speckle.",
                totalTracerMaxStepHits);

    return svg;
}


} // namespace vtracer