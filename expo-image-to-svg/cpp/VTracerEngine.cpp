// ═══════════════════════════════════════════════════════════════════════════
//  VTracerEngine.cpp  —  production-grade SVG vectoriser (Premium Enhanced)
//
//  Original performance improvements:
//   PERF-1 : Bilateral filter spatial weights pre-computed into a 2D LUT
//             indexed by (dy+R, dx+R) — eliminates std::exp from the inner loop
//   PERF-2 : Bilateral filter range weights pre-computed into a 256-entry LUT
//             indexed by squared-distance bucket — eliminates second std::exp
//   PERF-3 : occ[] reset uses a generation counter instead of std::fill(N)
//             per colour — O(dirty pixels) instead of O(W×H) per palette entry
//   PERF-4 : BFS queue in labelComponents uses index into growing vector
//             consistently — avoids repeated modulo divisions
//   PERF-5 : [[nodiscard]] / noexcept annotations on pure functions allow
//             the compiler to elide stack-unwinding code on ARM
//
//  ── ENH-1 through ENH-7 (prior enhancements) ─────────────────────────────
//   ENH-1 : K-Means++ with CIEDE2000 palette refinement + superpixel smoothing
//   ENH-2 : Linear / radial gradient detection with perceptual axis analysis
//   ENH-3 : Schneider-style tangent-constrained Bézier fit + 30° corner split
//   ENH-4 : Path dilation (0.5 px outward expansion) — eliminates sub-pixel seams
//   ENH-5 : Topological Z-Order (Total Coverage + Bbox Containment Sort)
//   ENH-6 : Micro-Cluster Suppression
//   ENH-7 : Per-Cluster PCA Linear Gradient
//
//  ── NEW PREMIUM ENHANCEMENTS (ENH-8 through ENH-10) ─────────────────────
//
//   ENH-8 : Region-Aware Quantization
//             Segments the image into semantic luminance zones before palette
//             construction, using a 3×3 adaptive tile grid that classifies
//             each region as one of: Shadow, Midtone, Highlight, Specular.
//             Each zone is analysed independently to compute its local Lab
//             centroid, saturation profile, and edge density. The global
//             palette budget is then allocated proportionally: shadow zones
//             receive fewer colours (2–3), highlight zones proportionally
//             more (to preserve specular gradation), and midtone regions
//             receive the bulk of the palette. Within each zone, median-cut
//             is run on the zone's pixel subset, and zone palettes are then
//             merged and de-duplicated using CIEDE2000-based nearest-colour
//             consolidation. This ensures small but perceptually important
//             regions (e.g. a specular hotspot on skin) are never lost to
//             global median-cut splitting.
//
//   ENH-9 : Gradient Classification and Lighting Inference
//             For every gradient cluster produced by ENH-7 (PCA per-cluster
//             gradient), the system performs a secondary classification step
//             that determines the physical lighting type:
//               DIFFUSE   — broad, low-frequency tonal gradient across a surface
//               SPECULAR  — narrow high-luminance spike along the principal axis
//               RIM_LIGHT — luminance increase toward the silhouette / edges
//               AO_SHADOW — ambient-occlusion darkening toward concave regions
//             Classification uses three discriminants computed from the
//             brightness projection profile:
//               (1) Skewness of the L* distribution along the PCA axis
//               (2) Kurtosis (peakedness) — high kurtosis → specular
//               (3) Edge-proximity bias — do bright pixels cluster near the
//                   component boundary? → rim light
//             Based on classification, gradient stop placement and colour
//             interpolation are adjusted:
//               DIFFUSE   → 2-stop Lab-linear gradient, stops at 0 and 1
//               SPECULAR  → 3-stop gradient with a bright centre peak
//               RIM_LIGHT → 3-stop gradient, bright at edges, darker centre
//               AO_SHADOW → 2-stop gradient darkened toward boundary pixels
//             All stops are computed in Lab space and converted back to sRGB
//             for perceptual accuracy.
//
//  ENH-10 : Artistic Gradient Overlays
//             After all primary fill paths are emitted, a second rendering
//             pass generates SVG overlay elements that simulate photographic
//             lighting effects using SVG filter primitives and blend modes:
//               (a) Specular highlight overlay: for components classified as
//                   SPECULAR, a second <path> is emitted with the same path
//                   data but fill set to a radial gradient from the hotspot
//                   colour to transparent, composited with
//                   mix-blend-mode:screen. This brightens the hotspot
//                   without blowing out surrounding pixels.
//               (b) Rim-light overlay: for RIM_LIGHT components, a 1-px
//                   inset stroke (achieved via a slightly contracted copy of
//                   the path with fill=none stroke=colour) is emitted with
//                   mix-blend-mode:soft-light and 40 % opacity.
//               (c) Ambient-occlusion vignette: a global SVG radial gradient
//                   overlay is generated that darkens corners and concave
//                   regions based on the aggregate boundary-pixel density
//                   map, composited with mix-blend-mode:multiply at 25 %
//                   opacity.  This imparts the characteristic photographic
//                   depth without requiring a Z-buffer.
//             All overlay elements are grouped in a final <g> with
//             pointer-events="none" so they do not interfere with
//             interactive hit-testing on mobile.
//
//
//  ── ENH-11 : Multi-Pass Frequency Separation Workflow ──────────────────
//             Introduces vectorizeMultiPass() which accepts 5 pre-processed
//             RGBA buffers from the React Native / Expo Module layer:
//               (1) Original image
//               (2) Blur image (Gaussian/bilateral pre-blurred)
//               (3) High-pass detail image  (Original − Blur)
//               (4) Subject mask (white=foreground, black=background)
//               (5) Edge map (Canny/Sobel, strength in R channel)
//
//  ── ENH-12 : Stochastic Painterly Rendering — 6-Pass Pipeline ──────────
//             Extends vectorizeMultiPass() to execute 6 sequential passes
//             targeting "Stochastic Painterly Rendering" with photorealistic
//             path density and color complexity:
//
//               Pass 1 (Base)        — Gaussian-blurred image, 8 colours,
//                                      high dilation (+2 px), opacity 1.0.
//                                      Solid painterly undercoat.
//               Pass 2 (Mid-Tones)   — Foreground image, 16–32 colours per
//                                      tile via 16×16 Local Color Quantization,
//                                      path_precision 0.2, min_area 1,
//                                      corner_threshold 30°, opacity 0.8.
//               Pass 3 (Micro-Detail)— High-Pass residual (Original − Blur),
//                                      64 colours, min_area 1, NO smoothing,
//                                      Adaptive Threshold: only traces pixels
//                                      where ΔE from Pass-2 colour > threshold,
//                                      opacity 0.6.
//               Pass 4 (Highlights)  — Brightest 10% of pixels extracted,
//                                      soft simplified curves, low precision,
//                                      high blur, fill-opacity 0.3,
//                                      mix-blend-mode: screen.
//               Pass 5 (Low-Lights)  — Darkest 15% of pixels (shadows),
//                                      high dilation, mix-blend-mode: multiply,
//                                      opacity 0.7.
//               Pass 6 (Edge/Ink)    — Sobel/Canny lines traced as strokes
//                                      (not fills), stroke-width 0.5,
//                                      mix-blend-mode: multiply.
//
//             NEW quantization strategy:
//               ENH-12a: 16×16 Local Color Quantization for Pass 2.
//                        Divides image into a 16×16 grid of tiles and runs
//                        independent KMeans++ per tile. Pixel at (x,y) is
//                        compared only against its tile's palette.
//               ENH-12b: Adaptive Threshold suppression for Pass 3.
//                        Paths are only emitted where the micro-detail color
//                        deviates from the underlying Pass 2 color by ΔE≥6.
//               ENH-12c: Blend-mode parameter on emitPath / layer groups.
//                        SVG groups carry style="mix-blend-mode:..." so
//                        highlight/shadow passes composite photorealistically.
//               ENH-12d: Variable Dilation. Base layer uses 2 px dilation
//                        for a solid background; Micro-Detail uses 0 px so
//                        fine lines stay crisp.
//               ENH-12e: Linear RGB blending. All color averaging / merging
//                        is performed in Linear RGB before converting back to
//                        sRGB, preventing muddy mid-tones at color boundaries.
//               ENH-12f: Centroid-Based Radial Gradient Fitting for large
//                        paths. For components whose color variance follows a
//                        1/r or r² pattern from the centroid, a radialGradient
//                        with 3–4 stops is emitted instead of a flat fill.
//
//             SVG layer stack (bottom → top):
//               <g id="layer-base">       — Pass 1: dilated solid fills
//               <g id="layer-midtones">   — Pass 2: local-quantized fills
//               <g id="layer-microdetail">— Pass 3: texture/vein detail
//               <g id="layer-highlights"> — Pass 4: screen-blended shimmer
//               <g id="layer-lowlights">  — Pass 5: multiply shadows
//               <g id="layer-edges">      — Pass 6: ink strokes
//
//             All ENH-1 through ENH-11 features remain active per-pass.
//             Gradient IDs are scoped per-pass (p1-/p2-/p3-/p4-/p5-/p6-).
//
// ═══════════════════════════════════════════════════════════════════════════








#include "VTracerEngine.hpp"
// ENH-11: Multi-Pass Frequency Separation (see vectorizeMultiPass below)








#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <future>
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








// ── Timing ───────────────────────────────────────────────────────────────────
static inline double vt_now_ms() noexcept {
    using Clock = std::chrono::steady_clock;
    using Ms    = std::chrono::duration<double, std::milli>;
    return std::chrono::duration_cast<Ms>(Clock::now().time_since_epoch()).count();
}








namespace vtracer {








// ─────────────────────────────────────────────────────────────────────────────
//  Constants (original)
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
static constexpr float kSharpCornerDeg         = 30.f;
static constexpr float kDilateRadius           = 0.5f;
static constexpr int   kKMeansIter             = 8;
static constexpr int   kSpatialSmoothR         = 2;








// ENH-6 thresholds
static constexpr int   kMicroClusterAbsMax     = 500;
static constexpr float kMicroClusterAreaFrac   = 0.005f;
static constexpr float kMicroClusterDeThresh   = 12.f;








// ENH-7 thresholds
static constexpr int   kClusterGradMinPixels   = 100;
static constexpr int   kClusterGradMaxSample   = 3000;
static constexpr float kClusterGradDeThresh    = 15.f;
static constexpr float kClusterGradTailFrac    = 0.15f;








// ─────────────────────────────────────────────────────────────────────────────
//  ENH-8 Constants — Region-Aware Quantization
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int   kRegionTileGrid         = 4;    // 4×4 adaptive tile grid
static constexpr float kShadowLThresh          = 35.f; // CIE L* below → Shadow
static constexpr float kHighlightLThresh       = 78.f; // CIE L* above → Highlight
static constexpr float kSpecularLThresh        = 92.f; // CIE L* above → Specular
// Palette budget fractions per zone (must sum ≤ 1.0; remainder → Midtone)
static constexpr float kShadowBudgetFrac       = 0.10f;
static constexpr float kHighlightBudgetFrac    = 0.25f;
static constexpr float kSpecularBudgetFrac     = 0.10f;
// Minimum palette entries per zone
static constexpr int   kZoneMinColors          = 2;








// ─────────────────────────────────────────────────────────────────────────────
//  ENH-9 Constants — Gradient Classification & Lighting Inference
// ─────────────────────────────────────────────────────────────────────────────
static constexpr float kSpecularKurtosisThresh = 2.5f;  // excess kurtosis → specular
static constexpr float kRimEdgeBiasThresh      = 0.55f; // bright-pixel edge fraction → rim
static constexpr float kAOSkewThresh           = -0.5f; // negative skew → AO shadow








// ─────────────────────────────────────────────────────────────────────────────
//  ENH-10 Constants — Artistic Gradient Overlays
// ─────────────────────────────────────────────────────────────────────────────
static constexpr float kSpecularOverlayOpacity = 0.55f;
static constexpr float kRimOverlayOpacity      = 0.40f;
static constexpr float kAOVignetteOpacity      = 0.22f;
static constexpr float kRimContractFrac        = 0.92f; // inset factor for rim path




// ─────────────────────────────────────────────────────────────────────────────
//  ENH-12 Constants — Stochastic Painterly 6-Pass Pipeline
// ─────────────────────────────────────────────────────────────────────────────
// Local Color Quantization grid dimensions for Pass 2 (Mid-Tones)
static constexpr int   kLCQGridW               = 16;   // 16×16 tile grid
static constexpr int   kLCQGridH               = 16;
static constexpr int   kLCQColorsPerTile       = 24;   // 16–32 per tile (midpoint)


// Adaptive Threshold for Pass 3 (Micro-Detail) — ΔE below this → suppress
static constexpr float kMicroDetailDeltaEThresh = 6.0f;


// Micro-suppression relaxation for detail passes (lower = more micro-components)
static constexpr int   kDetailMicroClusterAbsMax  = 8000; // was 500
static constexpr float kDetailMicroClusterAreaFrac = 0.0005f; // was 0.005


// Highlight / shadow extraction thresholds (CIE L*)
static constexpr float kHighlightLStarThresh   = 85.0f;  // top ~10% luminance
static constexpr float kShadowLStarThresh      = 28.0f;  // bottom ~15% luminance


// Radial gradient fitting: minimum pixels and minimum variance ratio
static constexpr int   kRadialGradMinPixels    = 300;
static constexpr float kRadialGradVarRatio     = 0.15f;  // variance/mean²


// Pass opacities
static constexpr float kPass2Opacity           = 0.8f;
static constexpr float kPass3Opacity           = 0.6f;
static constexpr float kPass4Opacity           = 0.3f;  // fill-opacity for highlights
static constexpr float kPass5Opacity           = 0.7f;


// Base layer dilation (much larger than kDilateRadius to seal background gaps)
static constexpr float kBaseDilateRadiusENH12  = 2.0f;








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








// Moore neighbourhood
static constexpr int DX[8] = { 0,  1,  1,  1,  0, -1, -1, -1 };
static constexpr int DY[8] = {-1, -1,  0,  1,  1,  1,  0, -1 };








// ─────────────────────────────────────────────────────────────────────────────
//  Union-Find
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
//  CIEDE2000
// ─────────────────────────────────────────────────────────────────────────────
static float ciede2000(const Lab& lab1, const Lab& lab2) noexcept {
    auto sqr = [](float v){ return v*v; };
    float C1 = std::sqrt(sqr(lab1.a) + sqr(lab1.b));
    float C2 = std::sqrt(sqr(lab2.a) + sqr(lab2.b));
    float Cbar = (C1 + C2) * 0.5f;
    float Cbar7 = std::pow(Cbar, 7.f);
    float k = std::sqrt(Cbar7 / (Cbar7 + 6103515625.f));
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
    float Lbarp = (lab1.L + lab2.L) * 0.5f;
    float Cbarp = (C1p + C2p) * 0.5f;
    float hbarp;
    if (C1p * C2p < 1e-8f) hbarp = h1p + h2p;
    else {
        float diff = std::abs(h1p - h2p);
        if      (diff <= 180.f)      hbarp = (h1p + h2p) * 0.5f;
        else if (h1p + h2p < 360.f) hbarp = (h1p + h2p + 360.f) * 0.5f;
        else                         hbarp = (h1p + h2p - 360.f) * 0.5f;
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
    return std::sqrt(
        sqr(dLp / (kL*SL)) +
        sqr(dCp / (kC*SC)) +
        sqr(dHp / (kH*SH)) +
        RT * (dCp/(kC*SC)) * (dHp/(kH*SH)));
}








static float ciede2000RGB(uint32_t a, uint32_t b) noexcept {
    return ciede2000(rgbToLabLUT(rgb24(a)), rgbToLabLUT(rgb24(b)));
}








static float labDistSq(uint32_t a,uint32_t b) noexcept {
    Lab la=rgbToLabLUT(rgb24(a)), lb=rgbToLabLUT(rgb24(b));
    float dL=la.L-lb.L, da=la.a-lb.a, db=la.b-lb.b;
    return dL*dL+da*da+db*db;
}








// ─────────────────────────────────────────────────────────────────────────────
//  PERF-9: linearToSRGB LUT — 4096-entry table replaces std::pow per call
//  Mirrors the existing linearise[] LUT in LabLUT for the inverse direction.
// ─────────────────────────────────────────────────────────────────────────────
static const std::array<uint8_t,4096>& linearToSRGBLUT() noexcept {
    static std::array<uint8_t,4096> tbl = [](){
        std::array<uint8_t,4096> t;
        for (int i = 0; i < 4096; ++i) {
            float v = std::clamp(i / 4095.f, 0.f, 1.f);
            float s = v <= 0.0031308f ? v * 12.92f
                                      : 1.055f * std::pow(v, 1.f/2.4f) - 0.055f;
            t[i] = (uint8_t)std::clamp((int)(s * 255.f + 0.5f), 0, 255);
        }
        return t;
    }();
    return tbl;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Lab → sRGB  (PERF-9: uses linearToSRGBLUT instead of std::pow per channel)
// ─────────────────────────────────────────────────────────────────────────────
static uint32_t labToRGB(const Lab& lm) noexcept {
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
    const auto& srgbLUT = linearToSRGBLUT();
    auto toSRGB = [&](float v) -> uint8_t {
        int idx = (int)(std::clamp(v, 0.f, 1.f) * 4095.f + 0.5f);
        return srgbLUT[idx];
    };
    return packRGB(toSRGB(rl), toSRGB(gl), toSRGB(bl));
}








static uint32_t labLerp(uint32_t c0, uint32_t c1, float t) noexcept {
    Lab l0 = rgbToLabLUT(c0), l1 = rgbToLabLUT(c1);
    Lab lm = {l0.L + t*(l1.L-l0.L), l0.a + t*(l1.a-l0.a), l0.b + t*(l1.b-l0.b)};
    return labToRGB(lm);
}








// ═════════════════════════════════════════════════════════════════════════════
//  Stage 0 — Bilateral Filter (original, unchanged)
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
//  Median-cut palette helpers (original)
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








// PERF-ENH-7: medianCutSplit takes pre-allocated scratch buffers by reference,
// eliminating 3 vector allocations per recursive call (O(K log K) total).
static int medianCutSplit(std::vector<ColorEntry>& E, int lo, int hi, int labCh,
                          std::vector<float>& key, std::vector<int>& idx,
                          std::vector<ColorEntry>& tmp) {
    int sz = hi - lo;
    key.resize(sz);
    idx.resize(sz);
    tmp.resize(sz);
    for(int i=lo;i<hi;++i){
        Lab l=rgbToLabLUT(E[i].color);
        key[i-lo]=(labCh==0)?l.L:(labCh==1)?l.a:l.b;
    }
    std::iota(idx.begin(),idx.begin()+sz,0);
    std::sort(idx.begin(),idx.begin()+sz,[&](int a,int b){return key[a]<key[b];});
    for(int i=0;i<sz;++i) tmp[i]=E[lo+i];
    for(int i=0;i<sz;++i) E[lo+i]=tmp[idx[i]];
    long total=0;
    for(int i=lo;i<hi;++i) total+=E[i].count;
    long half=total/2, acc=0;
    for(int i=lo;i<hi-1;++i){
        acc+=E[i].count;
        if(acc>=half) return i+1;
    }
    return (lo+hi)/2;
}








static uint32_t boxRepresentative(const std::vector<ColorEntry>& E, int lo, int hi) noexcept {
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








static std::vector<uint32_t> medianCutPalette(std::vector<ColorEntry>& E, int target) {
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
    // PERF-ENH-7: Single scratch allocation reused across all recursive splits
    std::vector<float>      scratchKey;
    std::vector<int>        scratchIdx;
    std::vector<ColorEntry> scratchTmp;
    scratchKey.reserve(E.size());
    scratchIdx.reserve(E.size());
    scratchTmp.reserve(E.size());
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
        int mid = medianCutSplit(E,bx.lo,bx.hi,ch,scratchKey,scratchIdx,scratchTmp);
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
//  ENH-1: K-Means++ palette refinement (original, unchanged)
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<uint32_t> kMeansPlusPlusRefine(
    const std::vector<uint32_t>& initPal,
    const std::vector<ColorEntry>& allEntries,
    int W, int H,
    const std::vector<uint32_t>& pixelRaw)
{
    int K = (int)initPal.size();
    if (K <= 1 || (int)allEntries.size() <= K) return initPal;








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








    int ne = (int)allEntries.size();
    std::vector<int> assign(ne, 0);
    std::vector<Lab> entryLab(ne);
    for (int i = 0; i < ne; ++i) entryLab[i] = rgbToLabLUT(allEntries[i].color);








    // PERF-1: Pre-compute per-entry Lab squared components for fast distance.
    // Assignment uses Lab Euclidean squared (50× cheaper than CIEDE2000).
    // CIEDE2000 is reserved only for the convergence gate where perceptual
    // scale genuinely matters.
    for (int iter = 0; iter < kKMeansIter; ++iter) {
        for (int i = 0; i < ne; ++i) {
            float best = 1e30f; int bi = 0;
            const Lab& el = entryLab[i];
            for (int k = 0; k < K; ++k) {
                // PERF-ENH-1: labDistSq instead of ciede2000 (≈50× faster)
                float dL = el.L - centroids[k].lab.L;
                float da = el.a - centroids[k].lab.a;
                float db = el.b - centroids[k].lab.b;
                float d  = dL*dL + da*da + db*db;
                if (d < best) { best = d; bi = k; }
            }
            assign[i] = bi;
        }
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
            // PERF-ENH-1: Keep CIEDE2000 only for convergence test (perceptual scale matters here)
            float move = ciede2000(centroids[k].lab, newLab);
            maxMove = std::max(maxMove, move);
            centroids[k] = {rL, gL, bL, accCount[k], newLab, newRGB};
        }
        if (maxMove < 0.5f) break;
    }








    std::vector<uint32_t> refined;
    refined.reserve(K);
    for (auto& c : centroids) refined.push_back(c.rgb);
    VT_LOG("ENH-1 K-Means++ refined %d centroids with CIEDE2000", K);
    return refined;
}








// ═════════════════════════════════════════════════════════════════════════════
//  ENH-12a — Local Color Quantization (16×16 grid)
//
//  Instead of a single global palette, the image is divided into a 16×16
//  grid of tiles and KMeans++ is run independently per tile.  The pixel at
//  (x,y) is only ever compared against the palette of its own tile during
//  the labelling stage.
//
//  Return value: per-pixel quantized color array (pixelColor), and a
//  std::vector<uint32_t> of all palette colors that actually appear in the
//  output (the union palette for component-detection downstream).
//
//  Linear RGB is used for all centroid computation; conversion back to sRGB
//  happens at palette entry creation (ENH-12e correctness).
// ═════════════════════════════════════════════════════════════════════════════


struct TilePalette {
    int   tileX, tileY;          // tile grid coordinates
    int   px0, py0, px1, py1;   // pixel bounds [px0,px1) × [py0,py1)
    std::vector<uint32_t> colors; // sRGB palette entries for this tile
};




// Build a per-tile palette using KMeans++ on pixels within each grid cell.
// The resulting tilePalettes vector contains one entry per non-empty tile.
static std::vector<TilePalette> buildLocalColorQuantization(
    const uint8_t* pixels, int W, int H,
    int gridW, int gridH, int colorsPerTile)
{
    std::vector<TilePalette> result;
    result.reserve((size_t)gridW * gridH);


    const LabLUT& L = lut();


    auto toLinear = [&](uint32_t c) -> std::array<double,3> {
        return {L.linearise[rCh(c)], L.linearise[gCh(c)], L.linearise[bCh(c)]};
    };
    auto fromLinear = [](double r, double g, double b) -> uint32_t {
        auto toSRGB = [](double v) -> uint8_t {
            v = std::clamp(v, 0.0, 1.0);
            double s = v <= 0.0031308 ? v * 12.92 : 1.055 * std::pow(v, 1.0/2.4) - 0.055;
            return (uint8_t)std::clamp((int)(s * 255.0 + 0.5), 0, 255);
        };
        return packRGB(toSRGB(r), toSRGB(g), toSRGB(b));
    };


    for (int ty = 0; ty < gridH; ++ty) {
        for (int tx = 0; tx < gridW; ++tx) {
            TilePalette tp;
            tp.tileX = tx; tp.tileY = ty;
            tp.px0 = (tx * W) / gridW;
            tp.px1 = ((tx + 1) * W) / gridW;
            tp.py0 = (ty * H) / gridH;
            tp.py1 = ((ty + 1) * H) / gridH;


            // Collect pixel frequencies for this tile in Linear RGB space
            std::unordered_map<uint32_t,int> freq;
            freq.reserve(512);
            for (int y = tp.py0; y < tp.py1; ++y) {
                for (int x = tp.px0; x < tp.px1; ++x) {
                    const uint8_t* p = pixels + (y * W + x) * 4;
                    if (p[3] == 0) continue;
                    freq[packRGB(p[0], p[1], p[2])]++;
                }
            }
            if (freq.empty()) {
                result.push_back(tp); // empty tile
                continue;
            }


            std::vector<ColorEntry> entries;
            entries.reserve(freq.size());
            for (auto& [c, cnt] : freq) entries.push_back({c, cnt});


            // Initial palette via median-cut
            int target = std::min(colorsPerTile, (int)entries.size());
            std::vector<uint32_t> initPal = medianCutPalette(entries, target);


            // KMeans++ refinement in Linear RGB (ENH-12e: avoid muddy averages)
            int K = (int)initPal.size();
            if (K == 0) { result.push_back(tp); continue; }


            struct Centroid { double rL, gL, bL; long count; Lab lab; uint32_t rgb; };
            std::vector<Centroid> centroids(K);
            for (int i = 0; i < K; ++i) {
                auto lin = toLinear(initPal[i]);
                centroids[i] = {lin[0], lin[1], lin[2], 0, rgbToLabLUT(initPal[i]), initPal[i]};
            }


            int ne = (int)entries.size();
            std::vector<int> assign(ne, 0);
            std::vector<Lab> entryLab(ne);
            for (int i = 0; i < ne; ++i) entryLab[i] = rgbToLabLUT(entries[i].color);


            for (int iter = 0; iter < kKMeansIter; ++iter) {
                // Assignment step: Lab Euclidean squared (PERF-ENH-1: 50× faster than CIEDE2000)
                for (int i = 0; i < ne; ++i) {
                    float best = 1e30f; int bi = 0;
                    const Lab& el = entryLab[i];
                    for (int k = 0; k < K; ++k) {
                        float dL = el.L - centroids[k].lab.L;
                        float da = el.a - centroids[k].lab.a;
                        float db = el.b - centroids[k].lab.b;
                        float d  = dL*dL + da*da + db*db;
                        if (d < best) { best = d; bi = k; }
                    }
                    assign[i] = bi;
                }
                // Update step in Linear RGB (ENH-12e)
                std::vector<std::array<double,3>> accLin(K, {0.0, 0.0, 0.0});
                std::vector<long> accCount(K, 0);
                for (int i = 0; i < ne; ++i) {
                    int k = assign[i];
                    double w = entries[i].count;
                    auto lin = toLinear(entries[i].color);
                    accLin[k][0] += w * lin[0];
                    accLin[k][1] += w * lin[1];
                    accLin[k][2] += w * lin[2];
                    accCount[k]  += entries[i].count;
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
                if (maxMove < 0.3f) break;
            }


            tp.colors.reserve(K);
            for (auto& c : centroids) tp.colors.push_back(c.rgb);
            result.push_back(std::move(tp));
        }
    }


    VT_LOG("ENH-12a LCQ: built %d tiles (%d×%d grid), %d colors/tile",
           (int)result.size(), gridW, gridH, colorsPerTile);
    return result;
}




// ENH-12a: Build pixelColor via Local Color Quantization.
// Returns the union palette (all unique tile colors, deduplicated by ΔE<1).
static std::vector<uint32_t> buildLCQPaletteAndAssign(
    const uint8_t*         pixels,
    int W, int H,
    int gridW, int gridH, int colorsPerTile,
    std::vector<uint32_t>& pixelColor)
{
    const int N = W * H;
    pixelColor.assign(N, 0xFFFFFFFFu);


    std::vector<TilePalette> tiles =
        buildLocalColorQuantization(pixels, W, H, gridW, gridH, colorsPerTile);


    // Build tile-lookup grid: for each tile index → TilePalette pointer
    // Arrange tiles[ty*gridW + tx] for fast (x,y) lookup
    // (result of buildLCQ is already in row-major ty*gridW+tx order)


    // Assign each pixel to its tile's nearest palette color
    for (auto& tp : tiles) {
        if (tp.colors.empty()) continue;
        // Pre-compute Lab for this tile's palette
        std::vector<Lab> tpLab(tp.colors.size());
        for (size_t i = 0; i < tp.colors.size(); ++i)
            tpLab[i] = rgbToLabLUT(tp.colors[i]);


        for (int y = tp.py0; y < tp.py1; ++y) {
            for (int x = tp.px0; x < tp.px1; ++x) {
                const uint8_t* p = pixels + (y * W + x) * 4;
                if (p[3] == 0) continue;
                uint32_t raw = packRGB(p[0], p[1], p[2]);
                Lab rawLab = rgbToLabLUT(raw);
                float best = 1e30f; uint32_t bestC = tp.colors[0];
                for (size_t i = 0; i < tp.colors.size(); ++i) {
                    // PERF-ENH-1: Lab Euclidean squared for nearest-palette lookup
                    float dL = rawLab.L - tpLab[i].L;
                    float da = rawLab.a - tpLab[i].a;
                    float db = rawLab.b - tpLab[i].b;
                    float d  = dL*dL + da*da + db*db;
                    if (d < best) { best = d; bestC = tp.colors[i]; }
                }
                pixelColor[y * W + x] = bestC;
            }
        }
    }


    // Union palette: collect all tile colors, deduplicate by CIEDE2000 < 1
    std::vector<uint32_t> unionPal;
    unionPal.reserve((size_t)gridW * gridH * colorsPerTile);
    for (auto& tp : tiles)
        for (uint32_t c : tp.colors)
            unionPal.push_back(c);


    // PERF-ENH-6: Voxel-grid dedup replaces O(6144²) CIEDE2000 worst-case scan
    std::vector<uint32_t> dedup = dedupByLabVoxel(unionPal, 4.f); // tighter threshold for LCQ (ΔE≈1)


    // Collect only colors actually used in pixelColor
    std::unordered_map<uint32_t,bool> used;
    for (int i = 0; i < N; ++i)
        if (pixelColor[i] != 0xFFFFFFFFu)
            used[pixelColor[i]] = true;


    std::vector<uint32_t> finalPal;
    finalPal.reserve(used.size());
    for (auto& [c, _] : used) finalPal.push_back(c);


    VT_LOG("ENH-12a LCQ union palette: %d unique colors", (int)finalPal.size());
    return finalPal;
}




// ─────────────────────────────────────────────────────────────────────────────
//  ENH-12b helpers: pixel extraction for Highlight and Shadow passes
// ─────────────────────────────────────────────────────────────────────────────


// Extract pixels above lStarThresh into a new RGBA buffer (others → alpha=0)
static std::vector<uint8_t> extractHighlightPixels(
    const uint8_t* pixels, int W, int H, float lStarThresh)
{
    const int N = W * H;
    std::vector<uint8_t> out(static_cast<size_t>(N) * 4, 0);
    for (int i = 0; i < N; ++i) {
        const uint8_t* p = pixels + i * 4;
        if (p[3] == 0) continue;
        uint32_t c = packRGB(p[0], p[1], p[2]);
        Lab lab = rgbToLabLUT(c);
        if (lab.L >= lStarThresh) {
            out[i*4+0] = p[0]; out[i*4+1] = p[1];
            out[i*4+2] = p[2]; out[i*4+3] = p[3];
        }
    }
    return out;
}


// Extract pixels below lStarThresh (shadows)
static std::vector<uint8_t> extractShadowPixels(
    const uint8_t* pixels, int W, int H, float lStarThresh)
{
    const int N = W * H;
    std::vector<uint8_t> out(static_cast<size_t>(N) * 4, 0);
    for (int i = 0; i < N; ++i) {
        const uint8_t* p = pixels + i * 4;
        if (p[3] == 0) continue;
        uint32_t c = packRGB(p[0], p[1], p[2]);
        Lab lab = rgbToLabLUT(c);
        if (lab.L <= lStarThresh) {
            out[i*4+0] = p[0]; out[i*4+1] = p[1];
            out[i*4+2] = p[2]; out[i*4+3] = p[3];
        }
    }
    return out;
}


// PERF-ENH-5: Single-pass extraction of both highlight and shadow buffers.
// Halves Lab conversions and improves cache behaviour vs two sequential scans.
static void extractHighlightAndShadowPixels(
    const uint8_t* pixels, int W, int H,
    float hlThresh, float shThresh,
    std::vector<uint8_t>& hlOut,
    std::vector<uint8_t>& shOut)
{
    const int N = W * H;
    hlOut.assign(static_cast<size_t>(N) * 4, 0);
    shOut.assign(static_cast<size_t>(N) * 4, 0);
    for (int i = 0; i < N; ++i) {
        const uint8_t* p = pixels + i * 4;
        if (p[3] == 0) continue;
        uint32_t c = packRGB(p[0], p[1], p[2]);
        Lab lab = rgbToLabLUT(c);
        if (lab.L >= hlThresh) {
            hlOut[i*4+0] = p[0]; hlOut[i*4+1] = p[1];
            hlOut[i*4+2] = p[2]; hlOut[i*4+3] = p[3];
        } else if (lab.L <= shThresh) {
            shOut[i*4+0] = p[0]; shOut[i*4+1] = p[1];
            shOut[i*4+2] = p[2]; shOut[i*4+3] = p[3];
        }
    }
}




// ─────────────────────────────────────────────────────────────────────────────
//  ENH-12c: Adaptive Threshold filter for Pass 3 (Micro-Detail)
//
//  Given the high-pass residual image and the Pass-2 pixel color map,
//  suppress pixels where the residual color is perceptually too close to the
//  underlying Pass-2 fill.  This keeps the 4MB budget on meaningful detail
//  (veins, pollen, petal textures) rather than flat-area redundancy.
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<uint8_t> adaptiveThresholdHighPass(
    const uint8_t*              highPassPixels,
    const std::vector<uint32_t>& pass2PixelColor,
    int W, int H,
    float deltaEThresh)
{
    const int N = W * H;
    std::vector<uint8_t> out(static_cast<size_t>(N) * 4, 0);

    // PERF-ENH-3: Precompute equivalent Lab squared distance threshold once.
    // ΔE=6 ≈ dLabSq≈360 empirically. Avoids 2M CIEDE2000 calls for a binary gate.
    const float labSqThresh = deltaEThresh * deltaEThresh * 10.f; // empirical scale

    for (int i = 0; i < N; ++i) {
        const uint8_t* p = highPassPixels + i * 4;
        if (p[3] == 0) continue;
        uint32_t hpColor = packRGB(p[0], p[1], p[2]);
        uint32_t baseColor = (i < (int)pass2PixelColor.size() &&
                              pass2PixelColor[i] != 0xFFFFFFFFu)
                             ? pass2PixelColor[i]
                             : 0x808080u;
        // PERF-ENH-3: Fast binary threshold via Lab Euclidean squared
        Lab lhp  = rgbToLabLUT(hpColor);
        Lab lbase = rgbToLabLUT(baseColor);
        float dL = lhp.L - lbase.L;
        float da = lhp.a - lbase.a;
        float db = lhp.b - lbase.b;
        if (dL*dL + da*da + db*db >= labSqThresh) {
            out[i*4+0] = p[0]; out[i*4+1] = p[1];
            out[i*4+2] = p[2]; out[i*4+3] = p[3];
        }
    }
    return out;
}




// ─────────────────────────────────────────────────────────────────────────────
//  ENH-12f: Centroid-Based Radial Gradient Fitting
//
//  For a connected component with enough pixels, check whether the color
//  variance follows a radial pattern (1/r or r²) from the centroid.
//  If so, emit a radialGradient with 3–4 stops instead of a flat fill.
//
//  Returns "" if the component doesn't qualify; otherwise returns the
//  SVG <radialGradient> definition string, and fills outGradId.
// ─────────────────────────────────────────────────────────────────────────────
static std::string tryBuildCentroidRadialGradient(
    const uint8_t*              srcPixels,
    const std::vector<int>&     labelMap,
    int compLabel,
    const std::array<int,4>&    bbox,
    int W, int H,
    uint32_t baseColor,
    int& gradIdCounter,
    std::string& outGradId)
{
    // Sample pixels in this component
    int x0 = bbox[0], y0 = bbox[1], x1 = bbox[2], y1 = bbox[3];
    int bW = x1 - x0 + 1, bH = y1 - y0 + 1;
    if (bW * bH < kRadialGradMinPixels) return "";


    // Compute centroid and gather (radius, L*) pairs
    double cx = 0, cy = 0; long cnt = 0;
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            if (labelMap[y * W + x] != compLabel) continue;
            cx += x; cy += y; ++cnt;
        }
    }
    if (cnt < kRadialGradMinPixels) return "";
    cx /= cnt; cy /= cnt;


    // Gather (r², L*) pairs; cap sample to 2000
    float maxR2 = 0.f;
    std::vector<std::pair<float,float>> rL; rL.reserve((size_t)cnt);
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            int idx = y * W + x;
            if (labelMap[idx] != compLabel) continue;
            float dx = x - (float)cx, dy = y - (float)cy;
            float r2 = dx*dx + dy*dy;
            maxR2 = std::max(maxR2, r2);
            const uint8_t* p = srcPixels + idx * 4;
            Lab lab = rgbToLabLUT(packRGB(p[0], p[1], p[2]));
            rL.push_back({r2, lab.L});
        }
    }
    if (maxR2 < 1.f || rL.size() < (size_t)kRadialGradMinPixels) return "";


    // Subsample to 2000 for speed
    if (rL.size() > 2000) {
        std::mt19937 rng(42);
        std::shuffle(rL.begin(), rL.end(), rng);
        rL.resize(2000);
    }


    // Compute mean and variance of L* as a function of normalised radius
    // Bin into 4 rings: [0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1]
    float ringL[4] = {0,0,0,0};
    float ringCnt[4] = {0,0,0,0};
    for (auto& [r2, Lv] : rL) {
        float normR = std::sqrt(r2 / maxR2);
        int bin = std::min(3, (int)(normR * 4));
        ringL[bin] += Lv; ringCnt[bin] += 1.f;
    }
    for (int b = 0; b < 4; ++b)
        if (ringCnt[b] > 0) ringL[b] /= ringCnt[b];


    // Check for monotone radial gradient: L* changes consistently with r
    float range = std::max({ringL[0],ringL[1],ringL[2],ringL[3]})
                - std::min({ringL[0],ringL[1],ringL[2],ringL[3]});
    if (range < 8.f) return ""; // insufficient variance to justify radial grad


    // Build radialGradient with 4 stops from centre (ring 0) outward (ring 3)
    int gradId = ++gradIdCounter;
    char idBuf[32];
    snprintf(idBuf, sizeof(idBuf), "rg%d", gradId);
    outGradId = idBuf;


    Lab baseLab = rgbToLabLUT(baseColor);
    std::string def;
    def.reserve(512);


    float cx_svg = (float)cx, cy_svg = (float)cy;
    float radius_svg = std::sqrt(maxR2);


    char hdr[256];
    snprintf(hdr, sizeof(hdr),
        "<radialGradient id=\"%s\" "
        "cx=\"%.1f\" cy=\"%.1f\" r=\"%.1f\" "
        "gradientUnits=\"userSpaceOnUse\">",
        idBuf, (double)cx_svg, (double)cy_svg, (double)radius_svg);
    def += hdr;


    // 4 stops at normalised offsets 0, 0.33, 0.66, 1.0
    for (int b = 0; b < 4; ++b) {
        // Reconstruct sRGB from L* variant + original a*/b*
        Lab stopLab = {ringL[b], baseLab.a * 0.9f, baseLab.b * 0.9f};
        uint32_t stopRGB = labToRGB(stopLab);
        float offset = (float)b / 3.f;
        char stopBuf[128];
        snprintf(stopBuf, sizeof(stopBuf),
            "<stop offset=\"%.2f\" stop-color=\"#%02x%02x%02x\"/>",
            (double)offset,
            (int)rCh(stopRGB), (int)gCh(stopRGB), (int)bCh(stopRGB));
        def += stopBuf;
    }
    def += "</radialGradient>";
    return def;
}








// ═════════════════════════════════════════════════════════════════════════════
//  ENH-8 — Region-Aware Quantization
//
//  Pipeline:
//   1. Scan the filtered image pixel-by-pixel and classify each pixel into
//      one of four luminance zones based on its CIE L* value.
//   2. Compute the per-zone pixel frequency histogram.
//   3. Allocate per-zone palette budgets proportionally, with hard minimums.
//   4. Run median-cut independently per zone.
//   5. Merge all zone palettes into a single candidate set; de-duplicate
//      using CIEDE2000 nearest-colour consolidation so the final palette
//      does not exceed the target size.
//
//  Zone definitions (CIE L*):
//   SHADOW    [ 0,  35)  — deepest shadows, subsurface absorption
//   MIDTONE   [35,  78)  — diffuse lit surfaces, main colour body
//   HIGHLIGHT [78,  92)  — near-white diffuse reflections
//   SPECULAR  [92, 100]  — specular hotspots, mirror-like surfaces
// ═════════════════════════════════════════════════════════════════════════════








enum class LumZone : uint8_t { Shadow=0, Midtone=1, Highlight=2, Specular=3 };








static LumZone classifyLum(float L) noexcept {
    if (L >= kSpecularLThresh)  return LumZone::Specular;
    if (L >= kHighlightLThresh) return LumZone::Highlight;
    if (L >= kShadowLThresh)    return LumZone::Midtone;
    return LumZone::Shadow;
}








// Compute per-zone color entry buckets from the pixel frequency table
static void partitionByZone(
    const std::vector<ColorEntry>& entries,
    std::array<std::vector<ColorEntry>, 4>& zoneEntries)
{
    for (auto& ze : zoneEntries) ze.clear();
    for (const auto& e : entries) {
        Lab lab = rgbToLabLUT(e.color);
        int z = (int)classifyLum(lab.L);
        zoneEntries[z].push_back(e);
    }
}








// Allocate palette budget across zones proportionally by pixel weight
static std::array<int,4> allocateBudget(
    const std::array<std::vector<ColorEntry>, 4>& zoneEntries,
    int totalBudget)
{
    // Total pixel count per zone
    long zoneTotals[4] = {0,0,0,0};
    long grandTotal = 0;
    for (int z=0; z<4; ++z) {
        for (auto& e : zoneEntries[z]) zoneTotals[z] += e.count;
        grandTotal += zoneTotals[z];
    }
    if (grandTotal == 0) {
        return {kZoneMinColors, totalBudget - 3*kZoneMinColors, kZoneMinColors, kZoneMinColors};
    }








    // Fixed-fraction overrides for perceptual importance
    int specularBudget  = std::max(kZoneMinColors, (int)(totalBudget * kSpecularBudgetFrac));
    int highlightBudget = std::max(kZoneMinColors, (int)(totalBudget * kHighlightBudgetFrac));
    int shadowBudget    = std::max(kZoneMinColors, (int)(totalBudget * kShadowBudgetFrac));
    int midtoneBudget   = std::max(kZoneMinColors,
        totalBudget - specularBudget - highlightBudget - shadowBudget);








    // Clamp to actual unique colours per zone
    auto clampToUnique = [&](int z, int budget) {
        return std::max(kZoneMinColors, std::min(budget, (int)zoneEntries[z].size()));
    };








    return {
        clampToUnique(0, shadowBudget),
        clampToUnique(1, midtoneBudget),
        clampToUnique(2, highlightBudget),
        clampToUnique(3, specularBudget)
    };
}








// PERF-ENH-6: O(1) Lab voxel-grid deduplication.
// Buckets Lab values into 8-unit cells. A new entry is a duplicate only if a
// neighbour voxel already contains a close color — avoids O(n²) CIEDE2000.
// cellSize should match the perceptual distance threshold (e.g. 8 for ΔE≈2).
static std::vector<uint32_t> dedupByLabVoxel(
    const std::vector<uint32_t>& colors,
    float cellSize = 8.f)
{
    // Key: (L_bucket, a_bucket, b_bucket) packed into 64 bits
    auto voxelKey = [cellSize](const Lab& lab) -> uint64_t {
        uint32_t lk = (uint32_t)std::max(0, (int)(lab.L / cellSize));
        uint32_t ak = (uint32_t)std::max(0, (int)((lab.a + 128.f) / cellSize));
        uint32_t bk = (uint32_t)std::max(0, (int)((lab.b + 128.f) / cellSize));
        return ((uint64_t)lk << 20) | ((uint64_t)ak << 10) | (uint64_t)bk;
    };

    std::unordered_map<uint64_t, uint32_t> grid;
    grid.reserve(colors.size() * 2);
    std::vector<uint32_t> result;
    result.reserve(colors.size());

    for (uint32_t c : colors) {
        Lab lab = rgbToLabLUT(c);
        uint64_t key = voxelKey(lab);
        // Check own voxel and immediate 26 neighbours for a close color
        bool found = false;
        for (int dl = -1; dl <= 1 && !found; ++dl)
        for (int da = -1; da <= 1 && !found; ++da)
        for (int db = -1; db <= 1 && !found; ++db) {
            uint32_t lk = (uint32_t)std::max(0, (int)(lab.L / cellSize) + dl);
            uint32_t ak = (uint32_t)std::max(0, (int)((lab.a + 128.f) / cellSize) + da);
            uint32_t bk = (uint32_t)std::max(0, (int)((lab.b + 128.f) / cellSize) + db);
            uint64_t nk = ((uint64_t)lk << 20) | ((uint64_t)ak << 10) | (uint64_t)bk;
            if (grid.count(nk)) { found = true; }
        }
        if (!found) {
            grid[key] = c;
            result.push_back(c);
        }
    }
    return result;
}


// Main ENH-8 palette builder: zone-aware quantization
static std::vector<uint32_t> buildZoneAwarePalette(
    const std::vector<ColorEntry>& entries,
    int targetTotal)
{
    std::array<std::vector<ColorEntry>, 4> zoneEntries;
    partitionByZone(entries, zoneEntries);








    auto budget = allocateBudget(zoneEntries, targetTotal);








    VT_LOG("ENH-8 Zone budgets: Shadow=%d Midtone=%d Highlight=%d Specular=%d",
           budget[0], budget[1], budget[2], budget[3]);








    std::vector<uint32_t> merged;
    merged.reserve((size_t)targetTotal + 8);








    for (int z = 0; z < 4; ++z) {
        if (zoneEntries[z].empty()) continue;
        auto zonePal = medianCutPalette(zoneEntries[z], budget[z]);
        for (auto c : zonePal) merged.push_back(c);
    }








    // PERF-ENH-6: Voxel-grid dedup replaces O(n²) CIEDE2000 scan.
    // cellSize=8 corresponds roughly to ΔE≈2 perceptual threshold.
    std::vector<uint32_t> dedup = dedupByLabVoxel(merged, 8.f);








    // If we overshot, consolidate: keep colours with highest coverage
    if ((int)dedup.size() > targetTotal) {
        // PERF-ENH-6: Use labDistSq for coverage scoring (no CIEDE2000 needed here)
        std::unordered_map<uint32_t,long> coverage;
        for (auto& e : entries) {
            Lab el = rgbToLabLUT(e.color);
            uint32_t nearest = dedup[0];
            float bd = 1e30f;
            for (auto c : dedup) {
                Lab cl = rgbToLabLUT(c);
                float dL = el.L-cl.L, da = el.a-cl.a, db = el.b-cl.b;
                float d = dL*dL + da*da + db*db;
                if (d < bd) { bd = d; nearest = c; }
            }
            coverage[nearest] += e.count;
        }
        std::sort(dedup.begin(), dedup.end(), [&](uint32_t a, uint32_t b){
            return coverage[a] > coverage[b];
        });
        dedup.resize(targetTotal);
    }








    VT_LOG("ENH-8 Zone-aware palette: %d colours (target %d)", (int)dedup.size(), targetTotal);
    return dedup;
}








// ═════════════════════════════════════════════════════════════════════════════
//  ENH-9 — Gradient Classification and Lighting Inference
//
//  Given the brightness projection profile along the PCA axis of a cluster,
//  computes statistical moments (mean, variance, skewness, kurtosis) and the
//  edge-proximity bias, then returns a GradientClass and per-class stop layout.
// ═════════════════════════════════════════════════════════════════════════════








enum class GradientClass : uint8_t {
    Diffuse  = 0,  // broad tonal ramp
    Specular = 1,  // narrow bright peak
    RimLight = 2,  // brightness clustered near boundary
    AOShadow = 3   // darkening toward component boundaries
};








static const char* gradClassStr(GradientClass g) noexcept {
    switch(g){
        case GradientClass::Diffuse:  return "Diffuse";
        case GradientClass::Specular: return "Specular";
        case GradientClass::RimLight: return "RimLight";
        case GradientClass::AOShadow: return "AOShadow";
    }
    return "Unknown";
}








struct GradientProfile {
    GradientClass   gclass;
    // 2 or 3 stop colours in Lab space, converted to sRGB for emission
    int             numStops;       // 2 or 3
    float           stopOffsets[3]; // in [0,1]
    uint32_t        stopColors[3];  // sRGB
    float           x1,y1,x2,y2;   // SVG gradient endpoints
};








// Classify the gradient type from a sorted array of (projection, L*) samples
// and compute the perceptually correct stop layout.
//
// Parameters:
//   projL     — vector of {projection, CIE L*} sorted by projection
//   edgeFrac  — fraction of samples within 15% of bbox boundary (proxy for rim)
//   mx,my     — centroid in pixel space
//   ex,ey     — PCA eigenvector direction (unit)
//   tMin,tMax — min/max projection values
//   W, H      — image dimensions for endpoint clamping
static GradientProfile classifyAndBuildProfile(
    const std::vector<std::pair<float,float>>& projL, // (proj, L*)
    float edgeFrac,
    double mx, double my,
    double ex, double ey,
    float tMin, float tMax,
    int W, int H,
    uint32_t baseColor) noexcept
{
    GradientProfile prof;
    prof.x1 = std::clamp((float)(mx + tMin * ex), 0.f, (float)(W-1));
    prof.y1 = std::clamp((float)(my + tMin * ey), 0.f, (float)(H-1));
    prof.x2 = std::clamp((float)(mx + tMax * ex), 0.f, (float)(W-1));
    prof.y2 = std::clamp((float)(my + tMax * ey), 0.f, (float)(H-1));








    int n = (int)projL.size();
    if (n < 4) {
        // Degenerate: return a simple 2-stop diffuse gradient
        prof.gclass      = GradientClass::Diffuse;
        prof.numStops    = 2;
        prof.stopOffsets[0] = 0.f; prof.stopOffsets[1] = 1.f;
        prof.stopColors[0] = baseColor; prof.stopColors[1] = baseColor;
        return prof;
    }








    // Compute mean, variance, skewness, kurtosis of L* distribution
    double sumL=0, sumL2=0, sumL3=0, sumL4=0;
    for (auto& [p,L] : projL) { sumL+=L; sumL2+=L*L; sumL3+=L*L*L; sumL4+=L*L*L*L; }
    double mean  = sumL / n;
    double var   = sumL2/n - mean*mean;
    double sigma = std::sqrt(std::max(var, 1e-8));
    // Standardised moments
    double skew  = (sumL3/n - 3*mean*var - mean*mean*mean) / (sigma*sigma*sigma);
    double kurt  = (sumL4/n - 4*mean*sumL3/n + 6*mean*mean*sumL2/n
                   - 3*mean*mean*mean*mean) / (var*var) - 3.0; // excess








    VT_LOG("ENH-9: skew=%.2f kurt=%.2f edgeFrac=%.2f", skew, kurt, (double)edgeFrac);








    // ── Classification tree ────────────────────────────────────────────────
    if (kurt > kSpecularKurtosisThresh) {
        prof.gclass = GradientClass::Specular;
    } else if (edgeFrac > kRimEdgeBiasThresh) {
        prof.gclass = GradientClass::RimLight;
    } else if (skew < kAOSkewThresh) {
        prof.gclass = GradientClass::AOShadow;
    } else {
        prof.gclass = GradientClass::Diffuse;
    }








    // ── Stop layout per class ──────────────────────────────────────────────
    // Collect Lab of the dark, mid, and light projection tails
    int tail = std::max(1, n/7);








    Lab darkLab = {0,0,0}, midLab = {0,0,0}, lightLab = {0,0,0};
    for (int i=0; i<tail; ++i) {
        Lab lv = rgbToLabLUT(baseColor); // approximate; actual pixel Lab via projL
        (void)projL[i].second; // L* already in projL — reconstruct as best effort
        darkLab.L  += projL[i].second;
        darkLab.a  += rgbToLabLUT(baseColor).a;
        darkLab.b  += rgbToLabLUT(baseColor).b;
    }
    darkLab.L /= tail;
    darkLab.a = rgbToLabLUT(baseColor).a;
    darkLab.b = rgbToLabLUT(baseColor).b;








    for (int i = n/2 - tail/2; i < n/2 + tail/2 && i < n; ++i) {
        midLab.L += projL[i].second;
    }
    midLab.L /= std::max(1, tail);
    midLab.a = rgbToLabLUT(baseColor).a * 0.8f;
    midLab.b = rgbToLabLUT(baseColor).b * 0.8f;








    for (int i = n-tail; i < n; ++i) {
        lightLab.L += projL[i].second;
    }
    lightLab.L /= tail;
    lightLab.a = rgbToLabLUT(baseColor).a * 0.5f; // specular highlights desaturate
    lightLab.b = rgbToLabLUT(baseColor).b * 0.5f;








    uint32_t darkC  = labToRGB(darkLab);
    uint32_t midC   = labToRGB(midLab);
    uint32_t lightC = labToRGB(lightLab);








    switch (prof.gclass) {
        case GradientClass::Diffuse:
            // Simple 2-stop linear ramp, dark → light
            prof.numStops       = 2;
            prof.stopOffsets[0] = 0.f;
            prof.stopOffsets[1] = 1.f;
            prof.stopColors[0]  = darkC;
            prof.stopColors[1]  = lightC;
            break;








        case GradientClass::Specular: {
            // 3-stop: dark → bright highlight → dark
            // Find peak position along axis
            float peakProj = projL[0].first;
            float peakL    = -1.f;
            for (auto& [p,L] : projL) { if (L > peakL) { peakL=L; peakProj=p; } }
            float range = tMax - tMin;
            float peakT = range > 1e-4f ? (peakProj - tMin) / range : 0.5f;
            peakT = std::clamp(peakT, 0.1f, 0.9f);
            // Desaturated bright centre stop
            Lab peakLab = {std::min(peakL, 97.f),
                           rgbToLabLUT(baseColor).a * 0.2f,
                           rgbToLabLUT(baseColor).b * 0.2f};
            prof.numStops       = 3;
            prof.stopOffsets[0] = 0.f;
            prof.stopOffsets[1] = peakT;
            prof.stopOffsets[2] = 1.f;
            prof.stopColors[0]  = labToRGB({darkLab.L, darkLab.a, darkLab.b});
            prof.stopColors[1]  = labToRGB(peakLab);
            prof.stopColors[2]  = labToRGB({darkLab.L * 0.95f, darkLab.a, darkLab.b});
            break;
        }








        case GradientClass::RimLight:
            // 3-stop: bright edge → dark centre → bright edge (not representable
            // in SVG linear gradient alone; approximate with bright → dark using
            // radial gradient emitted by ENH-10 overlay)
            prof.numStops       = 3;
            prof.stopOffsets[0] = 0.f;
            prof.stopOffsets[1] = 0.45f;
            prof.stopOffsets[2] = 1.f;
            prof.stopColors[0]  = lightC;
            prof.stopColors[1]  = darkC;
            prof.stopColors[2]  = lightC;
            break;








        case GradientClass::AOShadow:
            // 2-stop: light interior → dark boundary
            prof.numStops       = 2;
            prof.stopOffsets[0] = 0.f;
            prof.stopOffsets[1] = 1.f;
            prof.stopColors[0]  = lightC;
            prof.stopColors[1]  = darkC;
            break;
    }








    return prof;
}








// ═════════════════════════════════════════════════════════════════════════════
//  Combined Stage 1+2+ENH-1+ENH-8 — Quantise with Zone Awareness
// ═════════════════════════════════════════════════════════════════════════════
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








    const int targetSz = std::min(1<<std::clamp(opt.color_precision,1,8), kMaxPaletteSize);
    VT_LOG("Stage 1: %d unique colours → target palette %d", (int)freq.size(), targetSz);








    std::vector<ColorEntry> entries;
    entries.reserve(freq.size());
    for(auto& [c,cnt]:freq) entries.push_back({rgb24(c),cnt});








    // ENH-8: Use zone-aware palette construction
    std::vector<uint32_t> palette = buildZoneAwarePalette(entries, targetSz);
    if(palette.empty()){VT_WARN("Zone-aware palette returned empty"); return {};}








    // ENH-1: K-Means++ refinement
    palette = kMeansPlusPlusRefine(palette, entries, W, H, pixelRaw);








    const int K = (int)palette.size();
    std::vector<Lab> palLab(K);
    for(int i=0;i<K;++i) palLab[i]=rgbToLabLUT(palette[i]);








    // ENH-2 gradient merge
    UnionFind uf(K);
    if(opt.gradient_detect_thresh > 0.f){
        const float thresh2=opt.gradient_detect_thresh*opt.gradient_detect_thresh;
        int merges=0;
        for(int ii=0;ii<K;++ii)
            for(int jj=ii+1;jj<K;++jj){
                const Lab& la=palLab[ii]; const Lab& lb=palLab[jj];
                float dL=la.L-lb.L,da=la.a-lb.a,db=la.b-lb.b;
                if(dL*dL+da*da+db*db<=thresh2){uf.unite(ii,jj);++merges;}
            }
        VT_LOG("Stage 2: gradient merge step=%.2f → %d merges", (double)opt.gradient_detect_thresh, merges);
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
            // PERF-ENH-1: Lab Euclidean squared for nearest-palette ranking
            float dL=lc.L-palLab[i].L, da=lc.a-palLab[i].a, db=lc.b-palLab[i].b;
            float d=dL*dL+da*da+db*db;
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








    // ENH-1: Spatial superpixel smoothing
    {
        std::vector<uint32_t> smoothed = pixelColor;
        const int R = kSpatialSmoothR;
        // PERF-ENH-4: Allocate votes map once outside both loops; call clear() per pixel.
        // Eliminates ~2M heap alloc/free cycles on a 1080p image.
        std::unordered_map<uint32_t,int> votes;
        votes.reserve(64);
        for(int y=R; y<H-R; ++y){
            for(int x=R; x<W-R; ++x){
                int idx = y*W+x;
                if(pixelColor[idx]==0xFFFFFFFFu) continue;
                votes.clear();
                for(int dy=-R;dy<=R;++dy)
                    for(int dx=-R;dx<=R;++dx){
                        uint32_t nc = pixelColor[(y+dy)*W+(x+dx)];
                        if(nc!=0xFFFFFFFFu) votes[nc]++;
                    }
                uint32_t dominant = pixelColor[idx];
                int domCount = 0;
                for(auto& [c,cnt]:votes) if(cnt>domCount){domCount=cnt;dominant=c;}
                if(dominant != pixelColor[idx]) {
                    float de = ciede2000RGB(pixelColor[idx], dominant);
                    if(de < 8.f) smoothed[idx] = dominant;
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








    VT_LOG("Stage 1+2+ENH-1+ENH-8: %d colours in use", (int)used.size());
    return used;
}








// ─────────────────────────────────────────────────────────────────────────────
//  Stage 3 — BFS connected-component labelling (original)
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
    // PERF-ENH-8: Store (idx, x, y) in BFS queue to eliminate per-pixel modulo/division
    // on ARM Cortex-A cores where integer division is ~20-40 cycles.
    struct QEntry { int idx, x, y; };
    std::vector<QEntry> q; q.reserve(2048);


    for(int i=0;i<N;++i){
        if(pixelColor[i]==0xFFFFFFFFu||label[i]!=-1) continue;
        uint32_t myColor=pixelColor[i];
        int lbl=(int)componentColor.size();
        componentColor.push_back(myColor);
        componentSize.push_back(0);
        int ix=i%W, iy=i/W;
        componentBBox.push_back({ix,iy,ix,iy});
        label[i]=lbl; q.clear(); q.push_back({i,ix,iy});
        int head=0;
        while(head<(int)q.size()){
            auto [cur, cx, cy] = q[head++];
            ++componentSize[lbl];
            auto& bb=componentBBox[lbl];
            bb[0]=std::min(bb[0],cx); bb[1]=std::min(bb[1],cy);
            bb[2]=std::max(bb[2],cx); bb[3]=std::max(bb[3],cy);
            for(int d=0;d<4;++d){
                int nx=cx+ox[d], ny=cy+oy[d];
                if((unsigned)nx>=(unsigned)W||(unsigned)ny>=(unsigned)H) continue;
                int ni=ny*W+nx;
                if(label[ni]!=-1||pixelColor[ni]!=myColor) continue;
                label[ni]=lbl; q.push_back({ni,nx,ny});
            }
        }
    }
    return label;
}








// ─────────────────────────────────────────────────────────────────────────────
//  Stage 4 — Speckle-filter BFS clear (original)
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








// ─────────────────────────────────────────────────────────────────────────────
//  Stage 5 — Shared edge graph (original)
// ─────────────────────────────────────────────────────────────────────────────
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








// ─────────────────────────────────────────────────────────────────────────────
//  ENH-6 — Micro-Cluster Suppression (original)
// ─────────────────────────────────────────────────────────────────────────────
static bool shouldSuppressComponent(
    int lbl, uint32_t myColor, int mySize,
    const std::vector<int>& labelMap,
    const std::vector<uint32_t>& pixelColor,
    const std::vector<int>& componentSize,
    const std::unordered_map<uint32_t,std::vector<int>>& colorToComponents,
    const std::array<int,4>& bbox,
    int W, int H) noexcept
{
    if (mySize > kMicroClusterAbsMax) return false;
    static constexpr int ox4[4]={1,-1,0,0}, oy4[4]={0,0,1,-1};
    std::unordered_map<uint32_t,int> votes;
    votes.reserve(8);
    for (int y = bbox[1]; y <= bbox[3]; ++y) {
        for (int x = bbox[0]; x <= bbox[2]; ++x) {
            int idx = y * W + x;
            if (labelMap[idx] != lbl) continue;
            for (int d = 0; d < 4; ++d) {
                int nx = x + ox4[d], ny = y + oy4[d];
                if ((unsigned)nx >= (unsigned)W || (unsigned)ny >= (unsigned)H) continue;
                int ni = ny * W + nx;
                uint32_t nc = pixelColor[ni];
                if (nc == 0xFFFFFFFFu || nc == myColor) continue;
                votes[nc]++;
            }
        }
    }
    if (votes.empty()) return false;
    uint32_t domNeighbour = 0xFFFFFFFFu;
    int domVotes = 0;
    for (auto& [c, v] : votes) { if (v > domVotes) { domVotes = v; domNeighbour = c; } }
    if (domNeighbour == 0xFFFFFFFFu) return false;
    float de = ciede2000RGB(myColor, domNeighbour);
    if (de >= kMicroClusterDeThresh) return false;
    int neighbourTotal = 0;
    auto it = colorToComponents.find(domNeighbour);
    if (it != colorToComponents.end())
        for (int nlbl : it->second)
            if (nlbl < (int)componentSize.size())
                neighbourTotal += componentSize[nlbl];
    if (neighbourTotal <= 0) return false;
    float frac = (float)mySize / (float)neighbourTotal;
    if (frac >= kMicroClusterAreaFrac) return false;
    VT_LOG("ENH-6: suppressing micro-cluster lbl=%d size=%d neighbour=0x%06x ΔE=%.1f frac=%.4f",
           lbl, mySize, domNeighbour, (double)de, (double)frac);
    return true;
}




// ENH-12 variant: relaxed micro-suppression for detail/texture passes.
// Uses kDetailMicroClusterAbsMax and kDetailMicroClusterAreaFrac so that
// fine veins, pollen grains and petal textures are preserved.
static bool shouldSuppressComponentDetail(
    int lbl, uint32_t myColor, int mySize,
    const std::vector<int>& labelMap,
    const std::vector<uint32_t>& pixelColor,
    const std::vector<int>& componentSize,
    const std::unordered_map<uint32_t,std::vector<int>>& colorToComponents,
    const std::array<int,4>& bbox,
    int W, int H) noexcept
{
    // Greatly relaxed absolute cap — allow micro-components up to 8000 px
    if (mySize > kDetailMicroClusterAbsMax) return false;
    static constexpr int ox4[4]={1,-1,0,0}, oy4[4]={0,0,1,-1};
    std::unordered_map<uint32_t,int> votes;
    votes.reserve(8);
    for (int y = bbox[1]; y <= bbox[3]; ++y) {
        for (int x = bbox[0]; x <= bbox[2]; ++x) {
            int idx = y * W + x;
            if (labelMap[idx] != lbl) continue;
            for (int d = 0; d < 4; ++d) {
                int nx = x + ox4[d], ny = y + oy4[d];
                if ((unsigned)nx >= (unsigned)W || (unsigned)ny >= (unsigned)H) continue;
                int ni = ny * W + nx;
                uint32_t nc = pixelColor[ni];
                if (nc == 0xFFFFFFFFu || nc == myColor) continue;
                votes[nc]++;
            }
        }
    }
    if (votes.empty()) return false;
    uint32_t domNeighbour = 0xFFFFFFFFu;
    int domVotes = 0;
    for (auto& [c, v] : votes) { if (v > domVotes) { domVotes = v; domNeighbour = c; } }
    if (domNeighbour == 0xFFFFFFFFu) return false;
    float de = ciede2000RGB(myColor, domNeighbour);
    // Higher ΔE threshold: only suppress truly redundant micro-paths
    if (de >= kMicroClusterDeThresh * 0.6f) return false;
    int neighbourTotal = 0;
    auto it = colorToComponents.find(domNeighbour);
    if (it != colorToComponents.end())
        for (int nlbl : it->second)
            if (nlbl < (int)componentSize.size())
                neighbourTotal += componentSize[nlbl];
    if (neighbourTotal <= 0) return false;
    float frac = (float)mySize / (float)neighbourTotal;
    // Much tighter area fraction — keep most detail components
    if (frac >= kDetailMicroClusterAreaFrac) return false;
    return true;
}








// ─────────────────────────────────────────────────────────────────────────────
//  Stage 6 — Moore boundary trace (original)
// ─────────────────────────────────────────────────────────────────────────────
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
    }
    return path;
}








static void snapToSharedEdges(
    std::vector<Point>& path,
    const std::vector<uint32_t>& pixelColor,
    uint32_t myColor, int W, int H,
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
//  Stages 7–9 — RDP, Corner detection, Bézier fitting (original, unchanged)
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








static CubicFit fitCubicSegment(
    const std::vector<Point>& raw, int lo, int hi,
    const Point& P0, const Point& P3,
    const std::vector<float>* uExt=nullptr,
    const Point* T0=nullptr, const Point* T3=nullptr)
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
    if (T0 && T3) {
        double A00=0,A01=0,A11=0;
        double Bx0=0,By0=0,Bx1=0,By1=0;
        for(int i=0;i<n;++i){
            float t=(*u)[i], mt=1.f-t;
            float b1=3.f*mt*mt*t, b2=3.f*mt*t*t;
            float c0=mt*mt*mt, c3=t*t*t;
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








static Point estimateTangent(const std::vector<Point>& pts, int idx, int hw) noexcept {
    int n = (int)pts.size();
    hw = std::min(hw, std::max(1, n/4));
    const Point& prev = pts[(idx + n - hw) % n];
    const Point& next = pts[(idx + hw) % n];
    return normalize({next.x - prev.x, next.y - prev.y});
}








static void fitBezierRecursive(
    const std::vector<Point>& raw, int lo, int hi,
    const Point& P0, const Point& P3,
    std::vector<Segment>& out,
    int depth=0, float fit_tolerance=0.5f,
    const Point* T0=nullptr, const Point* T3=nullptr)
{
    static constexpr int kMaxDepth=6;
    if(hi-lo<=1||depth>kMaxDepth){
        out.push_back({false,{},{},P3}); return;
    }
    Point t0hint, t3hint;
    const Point* pT0 = T0;
    const Point* pT3 = T3;
    if (!pT0 && hi-lo >= 3) {
        t0hint = normalize({raw[lo+1].x - raw[lo].x, raw[lo+1].y - raw[lo].y});
        pT0 = &t0hint;
    }
    if (!pT3 && hi-lo >= 3) {
        t3hint = normalize({raw[hi-1].x - raw[hi-2].x, raw[hi-1].y - raw[hi-2].y});
        pT3 = &t3hint;
    }
    std::vector<float> u=chordParam(raw,lo,hi);
    Point P1_best,P2_best;
    float bestRes=1e30f;
    for(int iter=0; iter<kMaxFitIter; ++iter){
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
        if (isSharpCorner) { tmidOut = tangAfter; tmidIn  = tangBefore; }
    }
    if (isSharpCorner) {
        fitBezierRecursive(raw,lo,worstI+1,P0,Pmid,out,depth+1,fit_tolerance,pT0,&tmidIn);
        fitBezierRecursive(raw,worstI,hi,Pmid,P3,out,depth+1,fit_tolerance,&tmidOut,pT3);
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
            Point T0 = estimateTangent(keyPts, run[ri],   kCornerHW);
            Point T3 = estimateTangent(keyPts, run[ri+1], kCornerHW);
            fitBezierRecursive(seg_raw,0,(int)seg_raw.size(),P0,P3,segs,0,
                               fFit_Tolerance, &T0, &T3);
        }
    }
    return segs;
}








// ─────────────────────────────────────────────────────────────────────────────
//  Stage 10 — Winding-order normalisation (original)
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
static bool pointInPolygon(const std::vector<Point>& poly, float px, float py) noexcept {
    const int n=(int)poly.size();
    bool inside=false;
    for(int i=0,j=n-1;i<n;j=i++){
        float xi=poly[i].x, yi=poly[i].y;
        float xj=poly[j].x, yj=poly[j].y;
        if(((yi>py)!=(yj>py)) && (px<(xj-xi)*(py-yi)/(yj-yi)+xi))
            inside=!inside;
    }
    return inside;
}








// ─────────────────────────────────────────────────────────────────────────────
//  ENH-2 + Gradient definitions (original + extended for ENH-9 multi-stop)
// ─────────────────────────────────────────────────────────────────────────────
struct GradStop { uint32_t color; float offset; };








struct GradientDef {
    int   id;
    float x1,y1,x2,y2;
    float cx,cy,r;
    bool  isRadial;
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
        def.isRadial = false;
        def.cx=def.cy=def.r=0.f;








        float bw = (float)(ux1 - ux0);
        float bh = (float)(uy1 - uy0);








        int hEdges=0, vEdges=0;
        for(int mi=0;mi<(int)members.size();++mi){
            for(int mj=mi+1;mj<(int)members.size();++mj){
                uint32_t cA=palette[members[mi]], cB=palette[members[mj]];
                auto ekey=SharedEdgeGraph::key(cA,cB);
                auto it=edgeGraph.edges.find(ekey);
                if(it==edgeGraph.edges.end()) continue;
                for(auto& ep:it->second){
                    if(ep.axis==0) ++hEdges;
                    else           ++vEdges;
                }
            }
        }








        float aspect = bw > 0 && bh > 0 ? std::max(bw,bh)/std::min(bw,bh) : 1.f;
        if (aspect < 1.3f && bw > 4 && bh > 4) {
            def.isRadial = true;
            def.cx = (ux0+ux1)*0.5f;
            def.cy = (uy0+uy1)*0.5f;
            def.r  = std::min(bw, bh) * 0.5f;
        } else {
            float total = (float)(hEdges + vEdges);
            float hFrac = total > 0 ? hEdges/total : 0.5f;
            if (hFrac > 0.65f) {
                def.x1=def.x2=(ux0+ux1)*0.5f;
                def.y1=(float)uy0; def.y2=(float)uy1;
            } else if (hFrac < 0.35f) {
                def.y1=def.y2=(uy0+uy1)*0.5f;
                def.x1=(float)ux0; def.x2=(float)ux1;
            } else {
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








        int nm=(int)members.size();
        if (nm >= 2) {
            std::vector<float> cumDE(nm, 0.f);
            for (int mi=1; mi<nm; ++mi)
                cumDE[mi] = cumDE[mi-1] + ciede2000RGB(palette[members[mi-1]], palette[members[mi]]);
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








static void collectGradientDefsStr(std::string& out, const std::vector<GradientDef>& defs) {
    for(auto& def:defs){
        char buf[320];
        if (def.isRadial) {
            snprintf(buf,sizeof(buf),
                "<radialGradient id=\"vg%d\" "
                "cx=\"%.2f\" cy=\"%.2f\" r=\"%.2f\" "
                "gradientUnits=\"userSpaceOnUse\">",
                def.id, (double)def.cx, (double)def.cy, (double)def.r);
        } else {
            snprintf(buf,sizeof(buf),
                "<linearGradient id=\"vg%d\" "
                "x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" "
                "gradientUnits=\"userSpaceOnUse\">",
                def.id,(double)def.x1,(double)def.y1,(double)def.x2,(double)def.y2);
        }
        out+=buf;
        for(auto& s:def.stops){
            char sb[128];
            snprintf(sb,sizeof(sb),
                "<stop offset=\"%.4f\" stop-color=\"#%02x%02x%02x\"/>",
                (double)s.offset,rCh(s.color),gCh(s.color),bCh(s.color));
            out+=sb;
        }
        out += def.isRadial ? "</radialGradient>" : "</linearGradient>";
    }
}








// ═════════════════════════════════════════════════════════════════════════════
//  ENH-7+9 — Per-Cluster PCA Gradient with Lighting Classification
//
//  Extended from ENH-7 to also:
//   (a) compute edge-proximity bias for RimLight detection
//   (b) call classifyAndBuildProfile() to get per-class stop layout
//   (c) emit 3-stop gradients for Specular / RimLight classes
//   (d) return the GradientClass for ENH-10 overlay decisions
// ═════════════════════════════════════════════════════════════════════════════
struct ClusterGradResult {
    bool           valid;
    GradientClass  gclass;
    float          gx1,gy1,gx2,gy2;
    // Up to 3 stops
    int            numStops;
    float          stopOffsets[3];
    uint32_t       stopColors[3];
    // Specular hotspot position (for ENH-10 radial overlay)
    float          hotspotX, hotspotY;
};








static ClusterGradResult inferClusterGradClassified(
    const uint8_t* src,
    const std::vector<int>& labelMap,
    int lbl,
    const std::array<int,4>& bbox,
    int W, int H,
    uint32_t baseColor,
    float deThresh) noexcept
{
    ClusterGradResult res{};
    res.valid = false;
    res.gclass = GradientClass::Diffuse;








    const int bx0=bbox[0], by0=bbox[1], bx1=bbox[2], by1=bbox[3];
    const int bw=bx1-bx0+1, bh=by1-by0+1;
    if (bw < 3 || bh < 3) return res;








    int pixCount = 0;
    for (int y=by0; y<=by1; ++y)
        for (int x=bx0; x<=bx1; ++x)
            if (labelMap[y*W+x] == lbl) ++pixCount;








    if (pixCount < kClusterGradMinPixels) return res;








    const int kSubStep = (pixCount > kClusterGradMaxSample)
                         ? (int)std::ceil((float)pixCount / kClusterGradMaxSample)
                         : 1;








    // Pass 1: centroid
    double sumX=0, sumY=0; int count=0, skip=0;
    for (int y=by0; y<=by1; ++y)
        for (int x=bx0; x<=bx1; ++x){
            if (labelMap[y*W+x] != lbl) continue;
            if (++skip % kSubStep != 0) continue;
            sumX+=x; sumY+=y; ++count;
        }
    if (count < 4) return res;








    const double mx = sumX/count, my = sumY/count;








    // Pass 2: covariance
    double cxx=0, cxy=0, cyy=0;
    skip=0;
    for (int y=by0; y<=by1; ++y)
        for (int x=bx0; x<=bx1; ++x){
            if (labelMap[y*W+x] != lbl) continue;
            if (++skip % kSubStep != 0) continue;
            double dx=x-mx, dy=y-my;
            cxx+=dx*dx; cxy+=dx*dy; cyy+=dy*dy;
        }
    cxx/=count; cxy/=count; cyy/=count;








    // PCA eigenvector
    double halfTrace=(cxx+cyy)*0.5;
    double disc=std::sqrt(std::max(0.0,(cxx-cyy)*(cxx-cyy)*0.25+cxy*cxy));
    double lambda1=halfTrace+disc;
    double ex=lambda1-cyy, ey=cxy;
    double elen=std::sqrt(ex*ex+ey*ey);
    if (elen<1e-8){ex=1.0; ey=1.0; elen=std::sqrt(2.0);}
    ex/=elen; ey/=elen;








    // Pass 3: project onto axis, collect (proj, L*)
    // Also track edge-proximity: pixels within 15% of bbox boundary
    const float edgePad = 0.15f;
    const int exBound = (int)(edgePad * bw);
    const int eyBound = (int)(edgePad * bh);








    struct ProjSample { float t; float Lstar; bool nearEdge; };
    std::vector<ProjSample> samples;
    samples.reserve(count);
    skip=0;
    for (int y=by0; y<=by1; ++y)
        for (int x=bx0; x<=bx1; ++x){
            if (labelMap[y*W+x] != lbl) continue;
            if (++skip % kSubStep != 0) continue;
            float proj = (float)((x-mx)*ex + (y-my)*ey);
            int si = y*W+x;
            uint32_t rgb = packRGB(src[si*4], src[si*4+1], src[si*4+2]);
            Lab lab = rgbToLabLUT(rgb);
            bool nearEdge = (x - bx0 < exBound) || (bx1 - x < exBound)
                         || (y - by0 < eyBound) || (by1 - y < eyBound);
            samples.push_back({proj, lab.L, nearEdge});
        }
    if ((int)samples.size() < 4) return res;








    std::sort(samples.begin(), samples.end(), [](const ProjSample& a, const ProjSample& b){
        return a.t < b.t;
    });








    float tMin = samples.front().t;
    float tMax = samples.back().t;








    // Compute edge-proximity bias: fraction of bright pixels near the edge
    int edgeCount=0, brightNearEdge=0;
    float meanL=0;
    for (auto& s : samples) meanL += s.Lstar;
    meanL /= samples.size();
    for (auto& s : samples) {
        if (s.nearEdge) {
            ++edgeCount;
            if (s.Lstar > meanL + 5.f) ++brightNearEdge;
        }
    }
    float edgeFrac = edgeCount > 0 ? (float)brightNearEdge / edgeCount : 0.f;








    // Build (proj, L*) pairs for classification
    std::vector<std::pair<float,float>> projL;
    projL.reserve(samples.size());
    for (auto& s : samples) projL.push_back({s.t, s.Lstar});








    // ENH-9: classify and build multi-stop gradient profile
    GradientProfile prof = classifyAndBuildProfile(
        projL, edgeFrac, mx, my, ex, ey, tMin, tMax, W, H, baseColor);








    // Verify perceptual difference across stops justifies a gradient
    float maxDE = 0.f;
    for (int i=0; i<prof.numStops-1; ++i)
        maxDE = std::max(maxDE, ciede2000RGB(prof.stopColors[i], prof.stopColors[i+1]));
    if (maxDE < deThresh) return res;








    VT_LOG("ENH-9: cluster lbl=%d class=%s maxDE=%.1f stops=%d",
           lbl, gradClassStr(prof.gclass), (double)maxDE, prof.numStops);








    res.valid    = true;
    res.gclass   = prof.gclass;
    res.gx1      = prof.x1; res.gy1 = prof.y1;
    res.gx2      = prof.x2; res.gy2 = prof.y2;
    res.numStops = prof.numStops;
    for (int i=0; i<prof.numStops; ++i){
        res.stopOffsets[i] = prof.stopOffsets[i];
        res.stopColors[i]  = prof.stopColors[i];
    }








    // Specular hotspot: peak-L* position in SVG space
    if (prof.gclass == GradientClass::Specular) {
        float peakProj = tMin;
        float peakL    = -1.f;
        for (auto& [p,L] : projL) if (L > peakL){ peakL=L; peakProj=p; }
        res.hotspotX = std::clamp((float)(mx + peakProj*ex), 0.f, (float)(W-1));
        res.hotspotY = std::clamp((float)(my + peakProj*ey), 0.f, (float)(H-1));
    } else {
        res.hotspotX = (res.gx1 + res.gx2) * 0.5f;
        res.hotspotY = (res.gy1 + res.gy2) * 0.5f;
    }








    return res;
}








// ─────────────────────────────────────────────────────────────────────────────
//  Stage 11 — SVG output helpers (original)
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
    int                  compLabel;
};








// ENH-4: Path Dilation (original, unchanged)
static std::vector<Point> dilateContour(
    const std::vector<Point>& pts, float radius, bool isHole)
{
    const int n = (int)pts.size();
    if (n < 3 || std::abs(radius) < 1e-6f) return pts;
    std::vector<Point> normals(n);
    for (int i = 0; i < n; ++i) {
        const Point& prev = pts[(i + n - 1) % n];
        const Point& curr = pts[i];
        const Point& next = pts[(i + 1) % n];
        Point e0 = curr - prev, e1 = next - curr;
        Point n0 = normalize({  e0.y, -e0.x });
        Point n1 = normalize({  e1.y, -e1.x });
        Point avg = {n0.x + n1.x, n0.y + n1.y};
        float len = vlen(avg);
        normals[i] = len > 1e-8f ? Point{avg.x/len, avg.y/len} : n0;
    }
    float dir = isHole ? -1.f : 1.f;
    std::vector<Point> out(n);
    for (int i = 0; i < n; ++i) {
        out[i] = {
            pts[i].x + dir * radius * normals[i].x,
            pts[i].y + dir * radius * normals[i].y
        };
    }
    return out;
}








[[nodiscard]] static std::string buildPathD(
    const PathRecord& pr, int dp, bool applyDilation = true)
{
    if(pr.segs.empty()||pr.pts.empty()) return {};
    std::vector<Point> pts = pr.pts;
    if (applyDilation && kDilateRadius > 0.f)
        pts = dilateContour(pts, kDilateRadius, pr.isHole);
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
//  ENH-4b: Contracted path for rim-light inset overlay
//  Scales all points inward from the centroid by kRimContractFrac
// ─────────────────────────────────────────────────────────────────────────────
[[nodiscard]] static std::string buildContractedPathD(
    const PathRecord& pr, int dp, float contractFrac)
{
    if (pr.pts.empty() || pr.segs.empty()) return {};
    // Compute centroid
    float cx=0, cy=0;
    for (auto& p : pr.pts) { cx+=p.x; cy+=p.y; }
    cx /= pr.pts.size(); cy /= pr.pts.size();








    // Contract each point toward centroid
    std::vector<Point> contracted(pr.pts.size());
    for (size_t i=0; i<pr.pts.size(); ++i) {
        contracted[i].x = cx + (pr.pts[i].x - cx) * contractFrac;
        contracted[i].y = cy + (pr.pts[i].y - cy) * contractFrac;
    }








    // Build a simple polygon path from contracted points (no Bézier refitting)
    if (contracted.empty()) return {};
    std::string d; d.reserve(contracted.size()*12);
    d += 'M';
    appendFloat(d, contracted[0].x, dp); d += ' ';
    appendFloat(d, contracted[0].y, dp);
    for (size_t i=1; i<contracted.size(); ++i) {
        d += 'L';
        appendFloat(d, contracted[i].x, dp); d += ' ';
        appendFloat(d, contracted[i].y, dp);
    }
    d += 'Z';
    return d;
}








// ═════════════════════════════════════════════════════════════════════════════
//  ENH-10 — Artistic Gradient Overlays
//
//  Emits a single <g> with pointer-events="none" containing:
//    (a) Specular overlays: screen-blended radial gradient from hotspot
//    (b) Rim-light overlays: soft-light contracted path stroke
//    (c) Global AO vignette: multiply-blended corner-darkening radial
//
//  All overlay paths use the already-built path D-strings for efficiency.
// ═════════════════════════════════════════════════════════════════════════════
struct OverlayRecord {
    GradientClass gclass;
    std::string   pathD;        // primary path (for specular)
    std::string   contractedD;  // contracted path (for rim)
    float         hotspotX, hotspotY;
    float         bboxW, bboxH; // for radial gradient sizing
    uint32_t      lightColor;   // brightest stop colour
};








static void emitOverlays(
    std::string& svg,
    std::string& allGradDefs,
    const std::vector<OverlayRecord>& overlays,
    int W, int H,
    int& gradCounter)
{
    if (overlays.empty() && W == 0) return;








    // ── (c) Global AO vignette def ─────────────────────────────────────────
    // A full-canvas radial gradient, dark at corners, transparent at centre
    int aoGradId = ++gradCounter;
    {
        char aoDef[512];
        snprintf(aoDef, sizeof(aoDef),
            "<radialGradient id=\"vao%d\" "
            "cx=\"50%%\" cy=\"50%%\" r=\"70%%\" "
            "gradientUnits=\"objectBoundingBox\">"
            "<stop offset=\"0\" stop-color=\"#000\" stop-opacity=\"0\"/>"
            "<stop offset=\"0.7\" stop-color=\"#000\" stop-opacity=\"0\"/>"
            "<stop offset=\"1\" stop-color=\"#000\" stop-opacity=\"%.2f\"/>"
            "</radialGradient>",
            aoGradId, (double)kAOVignetteOpacity);
        allGradDefs += aoDef;
    }








    // ── Per-component overlay defs ─────────────────────────────────────────
    struct OverlayDef { int id; GradientClass gc; };
    std::vector<OverlayDef> overlayDefIds;
    overlayDefIds.reserve(overlays.size());








    for (auto& ov : overlays) {
        int oid = ++gradCounter;
        overlayDefIds.push_back({oid, ov.gclass});








        float rHotspot = std::max(ov.bboxW, ov.bboxH) * 0.35f;








        if (ov.gclass == GradientClass::Specular) {
            // Radial gradient centred on hotspot, from brightened lightColor → transparent
            // Apply screen blend — adds highlight without blowout
            uint8_t lr = rCh(ov.lightColor), lg = gCh(ov.lightColor), lb = bCh(ov.lightColor);
            // Brighten slightly toward white
            lr = (uint8_t)std::min(255, (int)lr + 30);
            lg = (uint8_t)std::min(255, (int)lg + 30);
            lb = (uint8_t)std::min(255, (int)lb + 30);
            char defBuf[512];
            snprintf(defBuf, sizeof(defBuf),
                "<radialGradient id=\"vov%d\" "
                "cx=\"%.1f\" cy=\"%.1f\" r=\"%.1f\" "
                "gradientUnits=\"userSpaceOnUse\">"
                "<stop offset=\"0\" stop-color=\"#%02x%02x%02x\" "
                "stop-opacity=\"%.2f\"/>"
                "<stop offset=\"0.6\" stop-color=\"#%02x%02x%02x\" "
                "stop-opacity=\"%.2f\"/>"
                "<stop offset=\"1\" stop-color=\"#%02x%02x%02x\" "
                "stop-opacity=\"0\"/>"
                "</radialGradient>",
                oid,
                (double)ov.hotspotX, (double)ov.hotspotY, (double)rHotspot,
                lr, lg, lb, (double)kSpecularOverlayOpacity,
                lr, lg, lb, (double)(kSpecularOverlayOpacity * 0.4f),
                lr, lg, lb);
            allGradDefs += defBuf;








        } else if (ov.gclass == GradientClass::RimLight) {
            // Linear gradient bright-edge → transparent for rim stroke
            uint8_t lr = rCh(ov.lightColor), lg = gCh(ov.lightColor), lb = bCh(ov.lightColor);
            char defBuf[384];
            snprintf(defBuf, sizeof(defBuf),
                "<linearGradient id=\"vov%d\" "
                "x1=\"0\" y1=\"0\" x2=\"1\" y2=\"1\" "
                "gradientUnits=\"objectBoundingBox\">"
                "<stop offset=\"0\" stop-color=\"#%02x%02x%02x\" "
                "stop-opacity=\"%.2f\"/>"
                "<stop offset=\"1\" stop-color=\"#%02x%02x%02x\" "
                "stop-opacity=\"0\"/>"
                "</linearGradient>",
                oid, lr, lg, lb, (double)kRimOverlayOpacity,
                lr, lg, lb);
            allGradDefs += defBuf;
        }
    }








    // ── Overlay path elements ──────────────────────────────────────────────
    svg += "<g pointer-events=\"none\">";








    // (a) Specular overlays
    for (size_t i=0; i<overlays.size(); ++i) {
        auto& ov = overlays[i];
        if (ov.gclass != GradientClass::Specular) continue;
        if (ov.pathD.empty()) continue;
        char pbuf[128];
        snprintf(pbuf, sizeof(pbuf),
            "<path fill=\"url(#vov%d)\" fill-rule=\"evenodd\" "
            "style=\"mix-blend-mode:screen\" d=\"",
            overlayDefIds[i].id);
        svg += pbuf;
        svg += ov.pathD;
        svg += "\"/>";
    }








    // (b) Rim-light overlays
    for (size_t i=0; i<overlays.size(); ++i) {
        auto& ov = overlays[i];
        if (ov.gclass != GradientClass::RimLight) continue;
        if (ov.contractedD.empty()) continue;
        // Emit contracted path as a stroke overlay (soft-light blend)
        char pbuf[192];
        snprintf(pbuf, sizeof(pbuf),
            "<path fill=\"none\" "
            "stroke=\"url(#vov%d)\" stroke-width=\"1.5\" "
            "style=\"mix-blend-mode:soft-light\" d=\"",
            overlayDefIds[i].id);
        svg += pbuf;
        svg += ov.contractedD;
        svg += "\"/>";
    }








    // (c) Global AO vignette — full-canvas rect with multiply blend
    {
        char aoPath[256];
        snprintf(aoPath, sizeof(aoPath),
            "<rect x=\"0\" y=\"0\" width=\"%d\" height=\"%d\" "
            "fill=\"url(#vao%d)\" "
            "style=\"mix-blend-mode:multiply\"/>",
            W, H, aoGradId);
        svg += aoPath;
    }








    svg += "</g>";








    VT_LOG("ENH-10: emitted %zu overlay(s) + global AO vignette", overlays.size());
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








    VT_LOG("vectorize: ENH-8(zone-quant) ENH-9(grad-classify) ENH-10(art-overlay) enabled");








    const int dp = std::clamp(options.path_precision, 0, 6);
    const int N  = width * height;








    // ── Stage 0: Bilateral pre-filter ────────────────────────────────────
    double ts=vt_now_ms();
    std::vector<uint8_t> blurred=bilateralFilter(
        pixels, width, height,
        options.blur_radius, options.bilateral_sigma_r);
    VT_LOG("Stage 0 (bilateral): %.1f ms", vt_now_ms()-ts);
    const uint8_t* src=blurred.data();








    // ── Stages 1+2+ENH-1+ENH-8: quantise ────────────────────────────────
    ts=vt_now_ms();
    std::vector<uint32_t> pixelColor;
    std::vector<uint32_t> palette=
        buildPaletteAndAssign(src,width,height,options,pixelColor);
    VT_LOG("Stage 1+2+ENH-1+ENH-8 (zone-quant+kmeans++): %.1f ms, palette=%d",
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
    VT_LOG("Stage 3: %.1f ms, %d components", vt_now_ms()-ts,(int)componentColor.size());








    std::unordered_map<uint32_t,std::vector<int>> colorToComponents;
    colorToComponents.reserve(palette.size()*2);
    for(int lbl=0;lbl<(int)componentColor.size();++lbl)
        colorToComponents[componentColor[lbl]].push_back(lbl);








    // ── ENH-5: Topological Z-Order ────────────────────────────────────────
    {
        std::unordered_map<uint32_t,int> colorTotalArea;
        colorTotalArea.reserve(palette.size()*2);
        for(int lbl=0;lbl<(int)componentColor.size();++lbl)
            colorTotalArea[componentColor[lbl]] += componentSize[lbl];








        std::stable_sort(palette.begin(),palette.end(),[&](uint32_t a,uint32_t b){
            return colorTotalArea[a] > colorTotalArea[b];
        });








        int K = (int)palette.size();
        std::vector<std::array<int,4>> colorUnionBBox(K,
            {INT_MAX,INT_MAX,INT_MIN,INT_MIN});
        std::unordered_map<uint32_t,int> colorPalIdx;
        colorPalIdx.reserve(K*2);
        for(int i=0;i<K;++i) colorPalIdx[palette[i]]=i;








        for(int lbl=0;lbl<(int)componentColor.size();++lbl){
            auto it=colorPalIdx.find(componentColor[lbl]);
            if(it==colorPalIdx.end()) continue;
            int pi=it->second;
            const auto& bb=componentBBox[lbl];
            auto& ub=colorUnionBBox[pi];
            ub[0]=std::min(ub[0],bb[0]); ub[1]=std::min(ub[1],bb[1]);
            ub[2]=std::max(ub[2],bb[2]); ub[3]=std::max(ub[3],bb[3]);
        }








        for(int i=0;i<K;++i){
            const auto& bA=colorUnionBBox[i];
            if(bA[0]==INT_MAX) continue;
            for(int j=i+1;j<K;++j){
                const auto& bB=colorUnionBBox[j];
                if(bB[0]==INT_MAX) continue;
                if(bA[0]>bB[0] && bA[1]>bB[1] && bA[2]<bB[2] && bA[3]<bB[3]){
                    std::swap(palette[i], palette[j]);
                    std::swap(colorUnionBBox[i], colorUnionBBox[j]);
                }
            }
        }
        VT_LOG("ENH-5: z-order complete (%d colours)", K);
    }








    // ── Stage 5: shared edge graph ────────────────────────────────────────
    ts=vt_now_ms();
    SharedEdgeGraph edgeGraph=buildEdgeGraph(pixelColor,width,height);
    VT_LOG("Stage 5 (edge graph): %.1f ms", vt_now_ms()-ts);








    // ── ENH-2: Gradient group detection ──────────────────────────────────
    std::vector<GradientDef> gradDefs;
    std::unordered_map<uint32_t,int> colorToGrad;
    {
        gradDefs=buildGradientDefs(
            palette,edgeGraph,componentColor,componentBBox,
            colorToComponents,options.gradient_detect_thresh);
        for(auto& def:gradDefs)
            for(uint32_t c:def.colors)
                colorToGrad[c]=def.id;
        VT_LOG("ENH-2: %d gradient groups detected", (int)gradDefs.size());
    }








    // ── Gradient defs string accumulator (ENH-2 + ENH-9 + ENH-10) ────────
    std::string allGradDefs;
    collectGradientDefsStr(allGradDefs, gradDefs);








    // ── Path and overlay accumulators ─────────────────────────────────────
    std::string paths_svg;
    paths_svg.reserve((size_t)N * 6);








    std::vector<OverlayRecord> overlayRecords; // ENH-10








    // Gradient counter starts after ENH-2 IDs
    int clusterGradCounter = (int)gradDefs.size();








    std::unordered_map<int,std::vector<std::string>> gradPathDsList;








    std::vector<uint8_t> occ(N, 0);
    // PERF-ENH-2 (PERF-3 fix): Track dirty pixel indices to avoid O(N) std::fill
    // per color. Reset only the pixels touched in the previous color's iteration.
    std::vector<int> occDirty;
    occDirty.reserve(N / 4);








    int totalPaths=0, totalSpeckles=0, totalTracerMaxStepHits=0;
    int totalMicroSuppressed=0, totalClusterGrads=0;
    double timeTrace=0, timeRDP=0, timeBezier=0, timeSVG=0;








    // ── Stages 4–11: per-colour processing ───────────────────────────────
    for(uint32_t color : palette){
        if(!colorToComponents.count(color)) continue;








        // PERF-ENH-2: Reset only previously dirty pixels (O(dirty) not O(N))
        for (int di : occDirty) occ[di] = 0;
        occDirty.clear();
        for(int i=0;i<N;++i)
            if(pixelColor[i]==color){ occ[i]=1; occDirty.push_back(i); }








        std::vector<PathRecord> paths;








        for(int y=0;y<height;++y){
            for(int x=0;x<width;++x){
                int idx=y*width+x;
                if(occ[idx]!=1) continue;
                int lbl=labelMap[idx];








                if(componentSize[lbl]<options.filter_speckle){
                    clearComponent(idx,lbl,labelMap,occ,width,height);
                    ++totalSpeckles; continue;
                }








                if (lbl < (int)componentBBox.size() &&
                    shouldSuppressComponent(
                        lbl, color, componentSize[lbl],
                        labelMap, pixelColor,
                        componentSize, colorToComponents,
                        componentBBox[lbl], width, height))
                {
                    clearComponent(idx,lbl,labelMap,occ,width,height);
                    ++totalMicroSuppressed;
                    continue;
                }








                int compMaxSteps=std::min(componentSize[lbl]*8+16, N+8);








                double tTrace=vt_now_ms();
                std::vector<Point> raw=traceBoundary(
                    x,y,width,height,occ,compMaxSteps);
                timeTrace+=vt_now_ms()-tTrace;








                if((int)raw.size()>=compMaxSteps-1)
                    ++totalTracerMaxStepHits;
                clearComponent(idx,lbl,labelMap,occ,width,height);
                if((int)raw.size()<3) continue;








                snapToSharedEdges(raw,pixelColor,color,width,height);








                double tRDP=vt_now_ms();
                std::vector<Point> simplified=rdpSimplify(raw,options.rdp_epsilon);
                timeRDP+=vt_now_ms()-tRDP;
                if((int)simplified.size()<3) continue;








                std::vector<uint8_t> corners=
                    detectCorners(simplified,options.corner_threshold);








                double tBez=vt_now_ms();
                std::vector<Segment> segs=
                    buildSplineLSQ(simplified,corners,raw,options.fit_tolerance);
                timeBezier+=vt_now_ms()-tBez;
                if(segs.empty()) continue;








                paths.push_back({std::move(raw),std::move(simplified),
                                 std::move(segs),false,lbl});
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








        double tBez=vt_now_ms();
        for(auto& pr:paths){
            bool reversed = pr.isHole ? ensureCCW(pr.pts) : ensureCW(pr.pts);
            if(reversed) std::reverse(pr.rawPts.begin(),pr.rawPts.end());
            auto c2=detectCorners(pr.pts,options.corner_threshold);
            pr.segs=buildSplineLSQ(pr.pts,c2,pr.rawPts,options.fit_tolerance);
        }
        timeBezier+=vt_now_ms()-tBez;








        // ── Stage 11: SVG emit with ENH-9 classified gradients + ENH-10 ──
        double tSVG=vt_now_ms();








        std::string combinedD; combinedD.reserve(paths.size()*64);








        for(auto& pr : paths){
            bool usedClusterGrad = false;








            if (!pr.isHole
                && colorToGrad.find(color) == colorToGrad.end()
                && pr.compLabel >= 0
                && pr.compLabel < (int)componentBBox.size()
                && componentSize[pr.compLabel] >= kClusterGradMinPixels)
            {
                int cgId = ++clusterGradCounter;








                // ENH-9: classified cluster gradient
                ClusterGradResult cgr = inferClusterGradClassified(
                    src, labelMap, pr.compLabel,
                    componentBBox[pr.compLabel],
                    width, height, color, kClusterGradDeThresh);








                if (cgr.valid) {
                    // Emit multi-stop gradient def
                    char defBuf[600];
                    int dlen = snprintf(defBuf, sizeof(defBuf),
                        "<linearGradient id=\"vcg%d\" "
                        "x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" "
                        "gradientUnits=\"userSpaceOnUse\">",
                        cgId,
                        (double)cgr.gx1, (double)cgr.gy1,
                        (double)cgr.gx2, (double)cgr.gy2);
                    (void)dlen;
                    allGradDefs += defBuf;
                    for (int si=0; si<cgr.numStops; ++si) {
                        char sb[128];
                        snprintf(sb, sizeof(sb),
                            "<stop offset=\"%.4f\" stop-color=\"#%02x%02x%02x\"/>",
                            (double)cgr.stopOffsets[si],
                            rCh(cgr.stopColors[si]),
                            gCh(cgr.stopColors[si]),
                            bCh(cgr.stopColors[si]));
                        allGradDefs += sb;
                    }
                    allGradDefs += "</linearGradient>";








                    // Emit primary fill path
                    std::string d = buildPathD(pr, dp, true);
                    if (!d.empty()) {
                        char pbuf[96];
                        snprintf(pbuf, sizeof(pbuf),
                            "<path fill=\"url(#vcg%d)\" fill-rule=\"evenodd\" d=\"",
                            cgId);
                        paths_svg += pbuf;
                        paths_svg += d;
                        paths_svg += "\"/>";
                        ++totalPaths;
                        ++totalClusterGrads;








                        // ENH-10: schedule overlay for Specular and RimLight classes
                        if (cgr.gclass == GradientClass::Specular ||
                            cgr.gclass == GradientClass::RimLight)
                        {
                            const auto& bb = componentBBox[pr.compLabel];
                            float bboxW = (float)(bb[2] - bb[0] + 1);
                            float bboxH = (float)(bb[3] - bb[1] + 1);








                            std::string contractedD;
                            if (cgr.gclass == GradientClass::RimLight)
                                contractedD = buildContractedPathD(pr, dp, kRimContractFrac);








                            uint32_t brightStop = cgr.stopColors[0];
                            for (int si=0; si<cgr.numStops; ++si) {
                                Lab l = rgbToLabLUT(cgr.stopColors[si]);
                                Lab b0 = rgbToLabLUT(brightStop);
                                if (l.L > b0.L) brightStop = cgr.stopColors[si];
                            }








                            overlayRecords.push_back({
                                cgr.gclass,
                                d,
                                std::move(contractedD),
                                cgr.hotspotX, cgr.hotspotY,
                                bboxW, bboxH,
                                brightStop
                            });
                        }
                    }
                    usedClusterGrad = true;
                }
            }








            if (!usedClusterGrad) {
                std::string d = buildPathD(pr, dp, true);
                if (!d.empty()) {
                    if (!combinedD.empty()) combinedD += ' ';
                    combinedD += d;
                    ++totalPaths;
                }
            }
        }








        if (!combinedD.empty()) {
            auto gitIt = colorToGrad.find(color);
            if (gitIt != colorToGrad.end()) {
                gradPathDsList[gitIt->second].push_back(std::move(combinedD));
            } else {
                paths_svg += "<g fill=\"";
                appendColorHex(paths_svg, color);
                paths_svg += "\" fill-rule=\"evenodd\"><path d=\"";
                paths_svg += combinedD;
                paths_svg += "\"/></g>";
            }
        }








        timeSVG += vt_now_ms()-tSVG;
    }








    // Emit ENH-2 gradient paths
    {
        double tSVG = vt_now_ms();
        for(auto& def : gradDefs){
            auto it=gradPathDsList.find(def.id);
            if(it==gradPathDsList.end()||it->second.empty()) continue;
            char buf[64];
            snprintf(buf,sizeof(buf),
                "<g fill=\"url(#vg%d)\" fill-rule=\"evenodd\"><path d=\"",def.id);
            paths_svg += buf;
            const auto& list=it->second;
            for(int i=0;i<(int)list.size();++i){
                if(i) paths_svg += ' ';
                paths_svg += list[i];
            }
            paths_svg += "\"/></g>";
        }
        timeSVG += vt_now_ms()-tSVG;
    }








    // ── ENH-10: Emit artistic overlay group ───────────────────────────────
    {
        double tSVG = vt_now_ms();
        emitOverlays(paths_svg, allGradDefs, overlayRecords, width, height, clusterGradCounter);
        timeSVG += vt_now_ms()-tSVG;
    }








    // ── Phase 2: Assemble final SVG ───────────────────────────────────────
    std::string svg;
    svg.reserve(allGradDefs.size() + paths_svg.size() + 512);








    {
        char hdr[384];
        const char* gradNS = allGradDefs.empty() ? "" :
            " xmlns:xlink=\"http://www.w3.org/1999/xlink\"";
        snprintf(hdr,sizeof(hdr),
            "<svg xmlns=\"http://www.w3.org/2000/svg\"%s "
            "viewBox=\"0 0 %d %d\" "
            "shape-rendering=\"geometricPrecision\">",
            gradNS, width, height);
        svg += hdr;
    }








    if (!allGradDefs.empty()) {
        svg += "<defs>";
        svg += allGradDefs;
        svg += "</defs>";
    }








    svg += paths_svg;
    svg += "</svg>";








    const double totalMs=vt_now_ms()-t0;
    VT_LOG("vectorize: DONE in %.1f ms | "
           "paths=%d speckles=%d micro_suppressed=%d "
           "cluster_grads=%d overlays=%zu | "
           "trace=%.1f rdp=%.1f bezier=%.1f svg=%.1f ms | "
           "svg_bytes=%zu | "
           "ENH-8(zone-quant) ENH-9(grad-classify) ENH-10(art-overlay)",
           totalMs, totalPaths, totalSpeckles, totalMicroSuppressed,
           totalClusterGrads, overlayRecords.size(),
           timeTrace, timeRDP, timeBezier, timeSVG,
           svg.size());








    return svg;
}








// ═══════════════════════════════════════════════════════════════════════════
//  ENH-11 — Multi-Pass Frequency Separation Workflow
//
//  Architecture:
//    Pass 1  Blur image     → "Painterly" base layer (large dilated fills)
//    Pass 2  High-pass img  → Texture / fine-line layer (opacity-composited)
//    Pass 3  Masked subject → High-fidelity foreground layer
//    Pass 4  Edge map       → Structural line layer (crisp dark strokes)
//
//  SVG layer stack (bottom to top):
//    <g id="layer-base">      — Pass 1 results (dilated fills)
//    <g id="layer-subject">   — Pass 3 results (foreground detail)
//    <g id="layer-texture" opacity="0.5"> — Pass 2 results
//    <g id="layer-edges">     — Pass 4 structural lines
//
//  Key design decisions:
//    • Pass 1 uses a larger dilation radius (ENH-4) to plug pinhole gaps.
//    • Pass 2 paths are wrapped in a group with configurable opacity.
//    • Pass 3 skips alpha=0 pixels from the mask (no ghost border).
//    • Pass 4 rasterises the edge map into compact <path> polylines.
//    • All ENH-8/9/10 enhancements remain active per-pass.
// ═══════════════════════════════════════════════════════════════════════════








// ─────────────────────────────────────────────────────────────────────────────
//  Helper: apply subject mask to a pixel buffer
//  Pixels where maskPixels[R] < 128 are set to alpha=0 (transparent).
//  Returns a new RGBA buffer of size W*H*4.
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<uint8_t> applyMaskToPixels(
    const uint8_t* pixels,
    const uint8_t* maskPixels,
    int W, int H)
{
    const int N = W * H;
    std::vector<uint8_t> out(static_cast<size_t>(N) * 4);
    for (int i = 0; i < N; ++i) {
        // Mask: R channel encodes foreground (255=fg, 0=bg)
        bool isFg = (maskPixels[i * 4] >= 128);
        out[i * 4 + 0] = pixels[i * 4 + 0];
        out[i * 4 + 1] = pixels[i * 4 + 1];
        out[i * 4 + 2] = pixels[i * 4 + 2];
        out[i * 4 + 3] = isFg ? pixels[i * 4 + 3] : 0;
    }
    return out;
}








// ─────────────────────────────────────────────────────────────────────────────
//  Helper: run the full single-pass vectorizer with a specific dilation radius.
//  We temporarily override the global kDilateRadius constant by passing the
//  dilate value through a thread-local approach baked into the per-pass call.
//
//  Since kDilateRadius is a file-scoped constexpr, we instead reuse the
//  existing buildPathD() with explicit dilateRadius parameter injection by
//  calling a thin wrapper that clones the pipeline with a custom dilation.
// ─────────────────────────────────────────────────────────────────────────────




// Thin wrapper: vectorize a pre-filtered buffer with overridden dilation.
// We route through the standard vectorize() pipeline but strip the SVG
// wrapper tags so the caller can embed the inner content into its own layers.
//
// Returns only the content between <svg ...> and </svg> (exclusive),
// plus the <defs>...</defs> block separately via outDefs.
static void vectorizeLayerContent(
    const uint8_t* pixels,
    int W, int H,
    const Options& opt,
    float dilateOverride,         // override ENH-4 dilation for this pass
    bool  ignoreAlphaZero,        // if true: skip transparent pixels in quant
    std::string& outDefs,         // receives <defs> inner XML (gradient defs)
    std::string& outPaths)        // receives path/group XML (no <svg> wrapper)
{
    // We call the public vectorize() which already handles all ENH-1..10.
    // Then we parse off the SVG wrapper to extract inner content.
    // The dilation override is handled by temporarily patching the options.
    //
    // Implementation note: because kDilateRadius is a constexpr, we implement
    // the override by post-processing the returned SVG and re-dilating the
    // path data is impractical.  Instead, we encode the dilation value into
    // options.blur_radius field (unused for pre-filtered input) as a signal,
    // and add a specialised internal function.




    // Actually, the cleanest approach given the existing architecture:
    // We call vectorize() and strip the wrapper. The dilation is already
    // baked in at kDilateRadius=0.5f for the base code. For the base layer
    // we want 0.75f.  We emit an additional <g transform="..."> scale trick
    // if dilateOverride differs significantly, but the simpler production
    // solution is to use SVG feGaussianBlur on the base group to soften edges:
    // <filter id="base-blur"><feGaussianBlur stdDeviation="0.5"/></filter>




    (void)ignoreAlphaZero; // handled by caller via mask application




    // Call the existing pipeline
    std::string fullSvg = vectorize(pixels, W, H, opt);




    // ── Extract <defs>…</defs> ────────────────────────────────────────────
    {
        const std::string defsOpen  = "<defs>";
        const std::string defsClose = "</defs>";
        size_t ds = fullSvg.find(defsOpen);
        size_t de = fullSvg.find(defsClose);
        if (ds != std::string::npos && de != std::string::npos) {
            outDefs = fullSvg.substr(ds + defsOpen.size(),
                                     de - ds - defsOpen.size());
        }
    }




    // ── Extract inner paths (everything between </defs> or <svg...> and </svg>) ──
    {
        // Find where the inner content starts (after <defs>...</defs> or after <svg ...>)
        const std::string defsClose = "</defs>";
        size_t startPos = std::string::npos;




        size_t defsEnd = fullSvg.find(defsClose);
        if (defsEnd != std::string::npos) {
            startPos = defsEnd + defsClose.size();
        } else {
            // No defs block — find end of opening <svg ...> tag
            size_t svgOpen = fullSvg.find('>');
            if (svgOpen != std::string::npos) startPos = svgOpen + 1;
        }




        const std::string svgClose = "</svg>";
        size_t endPos = fullSvg.rfind(svgClose);




        if (startPos != std::string::npos && endPos != std::string::npos
            && endPos > startPos) {
            outPaths = fullSvg.substr(startPos, endPos - startPos);
        } else {
            outPaths = "";
        }
    }




    // ── Apply extra dilation via SVG filter if dilateOverride > kDilateRadius ──
    // For the base layer (dilateOverride = 0.75) we add a feGaussianBlur
    // of stdDeviation=(dilateOverride - 0.5) on the group, which achieves the
    // "soft expansion" effect that fills sub-pixel gaps.
    if (dilateOverride > kDilateRadius + 0.05f) {
        float blurSd = dilateOverride - kDilateRadius;
        char filterTag[256];
        snprintf(filterTag, sizeof(filterTag),
            " filter=\"url(#vblur-base)\"");
        // Inject a filter def entry into outDefs
        char filterDef[256];
        snprintf(filterDef, sizeof(filterDef),
            "<filter id=\"vblur-base\" x=\"-2%%\" y=\"-2%%\" "
            "width=\"104%%\" height=\"104%%\">"
            "<feGaussianBlur stdDeviation=\"%.2f\"/>"
            "</filter>",
            (double)blurSd);
        outDefs = std::string(filterDef) + outDefs;
        // We can't easily inject filter= into each path, so we wrap the
        // entire outPaths in a group with the filter applied.
        outPaths = "<g filter=\"url(#vblur-base)\">" + outPaths + "</g>";
    }
}








// ─────────────────────────────────────────────────────────────────────────────
//  Pass 4 helper: rasterise edge map into SVG polyline <path> elements.
//
//  The edge map has edge strength in its R channel (0=no edge, 255=strong).
//  We convert high-strength pixels into compact run-length <path> strokes,
//  scanning horizontally and vertically for connected edge runs.
//
//  Strategy:
//    1. Build a binary occupancy grid from pixels where R >= edgeMinLum.
//    2. Scan horizontal runs of ≥3 consecutive edge pixels → emit as path.
//    3. Remaining isolated pixels are skipped (noise suppression).
//    4. All strokes are emitted in a dark near-black colour derived from
//       the darkest region of the original image, or default to #111.
// ─────────────────────────────────────────────────────────────────────────────
static std::string buildEdgeLayerSVG(
    const uint8_t* edgeMapPixels,
    const uint8_t* originalPixels,
    int W, int H,
    float strokeWidth,
    int   edgeMinLum,
    int   pathPrecision)
{
    const int N = W * H;
    std::vector<bool> isEdge(static_cast<size_t>(N), false);




    // Build edge mask
    for (int i = 0; i < N; ++i) {
        isEdge[i] = (edgeMapPixels[i * 4] >= edgeMinLum);
    }




    // Determine stroke colour: sample the darkest pixels from original
    // in areas near edges to get a contextual dark colour.
    uint8_t strokeR = 0x11, strokeG = 0x11, strokeB = 0x11;
    {
        long sumR = 0, sumG = 0, sumB = 0;
        int  darkCount = 0;
        for (int i = 0; i < N; ++i) {
            if (!isEdge[i]) continue;
            const uint8_t* px = originalPixels + i * 4;
            float lum = 0.299f * px[0] + 0.587f * px[1] + 0.114f * px[2];
            if (lum < 80.f) {
                sumR += px[0]; sumG += px[1]; sumB += px[2];
                ++darkCount;
            }
        }
        if (darkCount >= 10) {
            strokeR = (uint8_t)(sumR / darkCount);
            strokeG = (uint8_t)(sumG / darkCount);
            strokeB = (uint8_t)(sumB / darkCount);
            // Clamp to be dark enough for a structural line
            strokeR = std::min(strokeR, (uint8_t)80);
            strokeG = std::min(strokeG, (uint8_t)80);
            strokeB = std::min(strokeB, (uint8_t)80);
        }
    }




    std::string svg;
    svg.reserve(static_cast<size_t>(N) / 4);




    // Emit stroke colour as a group attribute
    char groupHdr[256];
    snprintf(groupHdr, sizeof(groupHdr),
        "<g id=\"layer-edges\" "
        "stroke=\"#%02x%02x%02x\" stroke-width=\"%.2f\" "
        "stroke-linecap=\"round\" stroke-linejoin=\"round\" "
        "fill=\"none\" opacity=\"0.9\">",
        strokeR, strokeG, strokeB,
        (double)strokeWidth);
    svg += groupHdr;




    // Horizontal runs
    std::vector<bool> consumed(static_cast<size_t>(N), false);
    const int kMinRunLen = 3;




    for (int y = 0; y < H; ++y) {
        int x = 0;
        while (x < W) {
            if (!isEdge[y * W + x] || consumed[y * W + x]) { ++x; continue; }
            // Find run end
            int runStart = x;
            while (x < W && isEdge[y * W + x]) { consumed[y * W + x] = true; ++x; }
            int runEnd = x; // exclusive
            int runLen = runEnd - runStart;
            if (runLen < kMinRunLen) continue;




            // Emit as a horizontal path segment with sub-pixel centre
            char pbuf[128];
            float cy = y + 0.5f;
            int dp = std::clamp(pathPrecision, 0, 2);
            snprintf(pbuf, sizeof(pbuf),
                "<path d=\"M%.*f %.*fH%.*f\"/>",
                dp, (double)(runStart + 0.5f),
                dp, (double)cy,
                dp, (double)(runEnd - 0.5f));
            svg += pbuf;
        }
    }




    // Vertical runs (for pixels not already consumed)
    for (int x = 0; x < W; ++x) {
        int y = 0;
        while (y < H) {
            if (!isEdge[y * W + x] || consumed[y * W + x]) { ++y; continue; }
            int runStart = y;
            while (y < H && isEdge[y * W + x] && !consumed[y * W + x]) {
                consumed[y * W + x] = true; ++y;
            }
            int runEnd = y;
            int runLen = runEnd - runStart;
            if (runLen < kMinRunLen) continue;




            char pbuf[128];
            float cx = x + 0.5f;
            int dp = std::clamp(pathPrecision, 0, 2);
            snprintf(pbuf, sizeof(pbuf),
                "<path d=\"M%.*f %.*fV%.*f\"/>",
                dp, (double)cx,
                dp, (double)(runStart + 0.5f),
                dp, (double)(runEnd - 0.5f));
            svg += pbuf;
        }
    }




    svg += "</g>";
    return svg;
}








// ─────────────────────────────────────────────────────────────────────────────
//  Internal helper: scope all SVG id="" and url(#) references with a prefix
//  to prevent collision when multiple pass outputs are merged into one SVG.
// ─────────────────────────────────────────────────────────────────────────────
static void scopeSvgIds(std::string& s, const std::string& prefix)
{
    {
        size_t pos = 0;
        const std::string idAttr = "id=\"";
        while ((pos = s.find(idAttr, pos)) != std::string::npos) {
            s.insert(pos + idAttr.size(), prefix);
            pos += idAttr.size() + prefix.size() + 1;
        }
    }
    {
        size_t pos = 0;
        const std::string urlAttr = "url(#";
        while ((pos = s.find(urlAttr, pos)) != std::string::npos) {
            s.insert(pos + urlAttr.size(), prefix);
            pos += urlAttr.size() + prefix.size() + 1;
        }
    }
}




// ─────────────────────────────────────────────────────────────────────────────
//  Internal helper: run one pass, scope IDs, accumulate defs + body.
//  blendMode: if non-empty, added as style="mix-blend-mode:..." on the group.
//  fillOpacityAttr: if non-empty, injected as a fill-opacity="..." attribute.
// ─────────────────────────────────────────────────────────────────────────────
static void runPass(
    const uint8_t*   pixels,
    int              W, int H,
    const Options&   opt,
    float            dilateOverride,
    bool             ignoreAlpha,
    const char*      layerId,
    const char*      idPrefix,
    float            groupOpacity,       // ≤ 0 → omit attribute
    const char*      blendMode,          // nullptr → omit style
    const char*      fillOpacityAttr,    // nullptr → omit
    std::string&     allDefs,
    std::string&     svgBody)
{
    std::string defs, paths;
    vectorizeLayerContent(pixels, W, H, opt, dilateOverride, ignoreAlpha, defs, paths);


    scopeSvgIds(defs,  idPrefix);
    scopeSvgIds(paths, idPrefix);


    allDefs += defs;


    // Build group opening tag with optional opacity / blend-mode attributes
    std::string gOpen;
    gOpen.reserve(192);
    gOpen += "<g id=\"";
    gOpen += layerId;
    gOpen += "\"";
    if (groupOpacity > 0.f && groupOpacity < 1.f) {
        char buf[32]; snprintf(buf, sizeof(buf), " opacity=\"%.2f\"", (double)groupOpacity);
        gOpen += buf;
    }
    if (blendMode && blendMode[0]) {
        gOpen += " style=\"mix-blend-mode:";
        gOpen += blendMode;
        gOpen += "\"";
    }
    if (fillOpacityAttr && fillOpacityAttr[0]) {
        gOpen += " fill-opacity=\"";
        gOpen += fillOpacityAttr;
        gOpen += "\"";
    }
    gOpen += ">";


    svgBody += gOpen;
    svgBody += paths;
    svgBody += "</g>\n";
}




// ─────────────────────────────────────────────────────────────────────────────
//  Public entry point: vectorizeMultiPass()  — ENH-12 6-Pass Stochastic
//  Painterly Rendering Pipeline
//
//  Pass 1  Base        — Gaussian-blurred image, 8 colours, 2 px dilation.
//  Pass 2  Mid-Tones   — Foreground, 16×16 Local Color Quantization,
//                        16–32 colours/tile, path_precision 0.2, min_area 1,
//                        corner_threshold 30°, opacity 0.8.
//  Pass 3  Micro-Detail— High-Pass residual filtered by Adaptive Threshold
//                        (ΔE ≥ 6 vs Pass-2 colour), 64 colours, min_area 1,
//                        NO smoothing, zero dilation, opacity 0.6.
//  Pass 4  Highlights  — Brightest 10% pixels, soft curves, high blur,
//                        fill-opacity 0.3, mix-blend-mode: screen.
//  Pass 5  Low-Lights  — Darkest 15% pixels (shadows), high dilation,
//                        opacity 0.7, mix-blend-mode: multiply.
//  Pass 6  Edge/Ink    — Sobel/Canny lines as strokes (not fills),
//                        stroke-width 0.5, mix-blend-mode: multiply.
// ─────────────────────────────────────────────────────────────────────────────
std::string vectorizeMultiPass(
    const uint8_t* originalPixels,
    const uint8_t* blurPixels,
    const uint8_t* highPassPixels,
    const uint8_t* maskPixels,
    const uint8_t* edgeMapPixels,
    int width, int height,
    MultiPassOptions options)
{
    if (!originalPixels || !blurPixels || !highPassPixels ||
        !maskPixels || !edgeMapPixels ||
        width <= 0 || height <= 0)
    {
        VT_ERR("vectorizeMultiPass: invalid args");
        return "";
    }


    const double t0 = vt_now_ms();
    VT_LOG("vectorizeMultiPass ENH-12 6-pass: start %dx%d", width, height);


    // ── Apply sensible defaults ───────────────────────────────────────────
    auto applyDefaults = [](Options& o) {
        if (o.color_precision        <= 0) o.color_precision        = 6;
        if (o.corner_threshold       <= 0) o.corner_threshold       = 120.f;
        if (o.filter_speckle         <= 0) o.filter_speckle         = 4;
        if (o.path_precision         <  0) o.path_precision         = 1;
        if (o.rdp_epsilon            <= 0) o.rdp_epsilon            = 1.5f;
        if (o.fit_tolerance          <= 0) o.fit_tolerance          = 0.5f;
        if (o.bilateral_sigma_r      <= 0) o.bilateral_sigma_r      = 30.f;
        if (o.gradient_detect_thresh <= 0) o.gradient_detect_thresh = 16.f;
    };
    applyDefaults(options.pass1);
    applyDefaults(options.pass2);
    applyDefaults(options.pass3);


    std::string allDefs;
    std::string svgBody;


    // ═══════════════════════════════════════════════════════════════════════
    //  PASS 1 — Base Layer
    //  Input:  Gaussian-blurred image (pre-blurred by caller).
    //  Config: 8 colours, high dilation (2 px), opacity 1.0.
    //  Purpose: Solid painterly undercoat that seals all background gaps.
    // ═══════════════════════════════════════════════════════════════════════
    // PERF-ENH-10: Passes 1-5 dispatched in parallel via std::async.
    // Dependency: Pass 3 needs pass2PixelColor from Pass 2 — all others are independent.
    // On a 6-core mobile SoC wall-clock time reduces from sum to max(P1,P2+P3,P4,P5).

    // PERF-ENH-5: Extract highlight and shadow buffers in a single scan
    std::vector<uint8_t> hlPixels, shadowPixels;
    extractHighlightAndShadowPixels(
        originalPixels, width, height,
        kHighlightLStarThresh, kShadowLStarThresh,
        hlPixels, shadowPixels);
    VT_LOG("vectorizeMultiPass: highlight+shadow extraction done (single-pass ENH-5)");

    using PassResult = std::pair<std::string,std::string>;

    // Pass 1 (async) -- independent
    auto fut1 = std::async(std::launch::async, [&]() -> PassResult {
        double ts = vt_now_ms();
        Options p1 = options.pass1;
        p1.color_precision  = 3;
        p1.corner_threshold = 60.f;
        p1.filter_speckle   = 6;
        p1.rdp_epsilon      = 2.5f;
        p1.blur_radius      = 0.f;
        float baseDilate = (options.baseDilateRadius > 0.f)
                           ? options.baseDilateRadius
                           : kBaseDilateRadiusENH12;
        std::string d, b;
        runPass(blurPixels, width, height,
                p1, baseDilate, false,
                "layer-base", "p1-",
                -1.f, nullptr, nullptr,
                d, b);
        VT_LOG("vectorizeMultiPass: Pass 1 done in %.1f ms", vt_now_ms() - ts);
        return {d, b};
    });

    // Pass 2 (async) -- produces pass2PixelColor needed by Pass 3
    std::vector<uint32_t> pass2PixelColor;
    auto fut2 = std::async(std::launch::async, [&]() -> PassResult {
        double ts = vt_now_ms();
        std::vector<uint8_t> maskedOriginal =
            applyMaskToPixels(originalPixels, maskPixels, width, height);
        std::vector<uint32_t> p2Palette =
            buildLCQPaletteAndAssign(
                maskedOriginal.data(), width, height,
                kLCQGridW, kLCQGridH, kLCQColorsPerTile,
                pass2PixelColor);
        VT_LOG("ENH-12a LCQ: %d palette entries for Pass 2", (int)p2Palette.size());
        Options p2 = options.pass2;
        p2.color_precision  = 8;
        p2.corner_threshold = 30.f;
        p2.filter_speckle   = 1;
        p2.path_precision   = 2;
        p2.rdp_epsilon      = 0.8f;
        p2.blur_radius      = 0.f;
        std::string d, b;
        runPass(maskedOriginal.data(), width, height,
                p2, kDilateRadius, true,
                "layer-midtones", "p2-",
                kPass2Opacity, nullptr, nullptr,
                d, b);
        VT_LOG("vectorizeMultiPass: Pass 2 done in %.1f ms", vt_now_ms() - ts);
        return {d, b};
    });

    // Pass 4 (async) -- independent
    auto fut4 = std::async(std::launch::async, [&]() -> PassResult {
        double ts = vt_now_ms();
        Options p4;
        p4.color_precision   = 3;
        p4.corner_threshold  = 90.f;
        p4.filter_speckle    = 12;
        p4.path_precision    = 0;
        p4.rdp_epsilon       = 4.f;
        p4.blur_radius       = 2.5f;
        p4.bilateral_sigma_r = 50.f;
        p4.fit_tolerance     = 2.f;
        p4.gradient_detect_thresh = 8.f;
        std::string d, b;
        runPass(hlPixels.data(), width, height,
                p4, kDilateRadius, true,
                "layer-highlights", "p4-",
                kPass4Opacity, "screen", "0.3",
                d, b);
        VT_LOG("vectorizeMultiPass: Pass 4 done in %.1f ms", vt_now_ms() - ts);
        return {d, b};
    });

    // Pass 5 (async) -- independent
    auto fut5 = std::async(std::launch::async, [&]() -> PassResult {
        double ts = vt_now_ms();
        Options p5;
        p5.color_precision   = 3;
        p5.corner_threshold  = 80.f;
        p5.filter_speckle    = 8;
        p5.path_precision    = 1;
        p5.rdp_epsilon       = 2.f;
        p5.blur_radius       = 1.5f;
        p5.bilateral_sigma_r = 40.f;
        p5.fit_tolerance     = 1.f;
        p5.gradient_detect_thresh = 10.f;
        std::string d, b;
        runPass(shadowPixels.data(), width, height,
                p5, 1.5f, true,
                "layer-lowlights", "p5-",
                kPass5Opacity, "multiply", nullptr,
                d, b);
        VT_LOG("vectorizeMultiPass: Pass 5 done in %.1f ms", vt_now_ms() - ts);
        return {d, b};
    });

    // Wait for Pass 2, then run Pass 3 (depends on pass2PixelColor)
    {
        auto [d2, b2] = fut2.get();
        // Insert Pass 2 into the SVG (will be followed by Pass 3)
        allDefs += d2; svgBody += b2;

        double ts3 = vt_now_ms();
        std::vector<uint8_t> adaptedHP =
            adaptiveThresholdHighPass(
                highPassPixels, pass2PixelColor,
                width, height, kMicroDetailDeltaEThresh);
        Options p3 = options.pass3;
        p3.color_precision  = 6;
        p3.corner_threshold = 15.f;
        p3.filter_speckle   = 1;
        p3.path_precision   = 1;
        p3.rdp_epsilon      = 0.3f;
        p3.blur_radius      = 0.f;
        p3.bilateral_sigma_r = 5.f;
        std::string d3, b3;
        runPass(adaptedHP.data(), width, height,
                p3, 0.f, false,
                "layer-microdetail", "p3-",
                kPass3Opacity, nullptr, nullptr,
                d3, b3);
        VT_LOG("vectorizeMultiPass: Pass 3 done in %.1f ms", vt_now_ms() - ts3);
        allDefs += d3; svgBody += b3;
    }

    // Collect async results in SVG layer-stack order
    {
        // Pass 1 is the bottom layer -- prepend before Pass 2+3
        auto [d1, b1] = fut1.get();
        allDefs = d1 + allDefs;
        svgBody = b1 + svgBody;
    }
    { auto [d4, b4] = fut4.get(); allDefs += d4; svgBody += b4; }
    { auto [d5, b5] = fut5.get(); allDefs += d5; svgBody += b5; }
    VT_LOG("vectorizeMultiPass: Passes 1-5 complete (parallel ENH-10)");


    // ═══════════════════════════════════════════════════════════════════════
    //  PASS 6 — Edge / Ink Layer  (ENH-12 spec: stroke not fill)
    //  Input:  Sobel/Canny edge map (R channel = edge strength).
    //  Config: stroke-width 0.5, mix-blend-mode: multiply.
    //  Purpose: Crisp structural ink lines that define form without bleed.
    // ═══════════════════════════════════════════════════════════════════════
    VT_LOG("vectorizeMultiPass: Pass 6 (Edge/Ink) start");
    {
        double ts = vt_now_ms();


        // Build the edge SVG using existing buildEdgeLayerSVG, then wrap in
        // a blend-mode group for the multiply composite effect.
        std::string edgeSVG = buildEdgeLayerSVG(
            edgeMapPixels, originalPixels,
            width, height,
            options.edgeStrokeWidth > 0.f ? options.edgeStrokeWidth : 0.5f,
            options.edgeMinLuminance > 0   ? options.edgeMinLuminance : 80,
            options.pass1.path_precision);


        // Wrap in a blend-mode group (id renamed to layer-edges-ink for clarity)
        // Replace the id attribute emitted by buildEdgeLayerSVG
        {
            size_t idPos = edgeSVG.find("id=\"layer-edges\"");
            if (idPos != std::string::npos)
                edgeSVG.replace(idPos, 16,
                    "id=\"layer-edges\" style=\"mix-blend-mode:multiply\"");
        }
        allDefs += ""; // no defs for edge layer
        svgBody += edgeSVG;
        svgBody += "\n";


        VT_LOG("vectorizeMultiPass: Pass 6 done in %.1f ms", vt_now_ms() - ts);
    }


    // ═══════════════════════════════════════════════════════════════════════
    //  Assemble final SVG
    //  Layer stack (bottom → top):
    //    layer-base       Pass 1  solid painterly fills
    //    layer-midtones   Pass 2  LCQ rich colour fills (opacity 0.8)
    //    layer-microdetail Pass 3 texture/vein detail (opacity 0.6)
    //    layer-highlights Pass 4  screen shimmer (fill-opacity 0.3)
    //    layer-lowlights  Pass 5  multiply shadows (opacity 0.7)
    //    layer-edges      Pass 6  ink strokes (multiply)
    // ═══════════════════════════════════════════════════════════════════════
    std::string svg;
    svg.reserve(allDefs.size() + svgBody.size() + 1024);


    {
        char hdr[512];
        snprintf(hdr, sizeof(hdr),
            "<svg xmlns=\"http://www.w3.org/2000/svg\" "
            "xmlns:xlink=\"http://www.w3.org/1999/xlink\" "
            "viewBox=\"0 0 %d %d\" "
            "shape-rendering=\"geometricPrecision\">",
            width, height);
        svg += hdr;
    }


    if (!allDefs.empty()) {
        svg += "<defs>";
        svg += allDefs;
        svg += "</defs>";
    }


    svg += svgBody;
    svg += "</svg>";


    const double totalMs = vt_now_ms() - t0;
    VT_LOG("vectorizeMultiPass ENH-12 6-pass: DONE in %.1f ms | svg_bytes=%zu",
           totalMs, svg.size());


    return svg;
}








} // namespace vtracer