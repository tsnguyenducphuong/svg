// ═══════════════════════════════════════════════════════════════════════════
//  VTracerEngine.cpp  —  production-grade SVG vectoriser
//
//  Pipeline:
//   Stage 0  Gaussian pre-blur          smooths pixel noise before tracing
//   Stage 1  Median-cut quantisation    perceptually accurate palette
//   Stage 2  CIE-Lab gradient merge     perceptual colour distance
//   Stage 3  BFS connected components   4-connectivity per colour
//   Stage 4  Speckle filter             drop tiny components
//   Stage 5  Moore boundary trace       Jacob's stopping criterion
//   Stage 6  RDP simplification         Ramer-Douglas-Peucker
//   Stage 7  Corner detection           angle-based per vertex
//   Stage 8  G1-continuous Bézier fit   Catmull-Rom with tangent fix-up
//   Stage 9  Winding-order fix          CW outer, CCW holes
//   Stage 10 SVG emission               relative cmds, short hex, <g> groups
// ═══════════════════════════════════════════════════════════════════════════

#include "VTracerEngine.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

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

// ───────────────────────────────────────────────────────────────────────────
//  Geometry types
// ───────────────────────────────────────────────────────────────────────────
struct Point   { float x, y; };
struct Segment { bool isCurve; Point cp1, cp2, end; };

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

// ── CIE-Lab perceptual colour distance ─────────────────────────────────────
struct Lab { float L, a, b; };

static float linearise(float u) {
    u /= 255.f;
    return (u <= 0.04045f) ? u / 12.92f : std::pow((u + 0.055f) / 1.055f, 2.4f);
}
static Lab rgbToLab(uint32_t c) {
    float rl = linearise((float)rCh(c));
    float gl = linearise((float)gCh(c));
    float bl = linearise((float)bCh(c));
    float X = rl*0.4124564f + gl*0.3575761f + bl*0.1804375f;
    float Y = rl*0.2126729f + gl*0.7151522f + bl*0.0721750f;
    float Z = rl*0.0193339f + gl*0.1191920f + bl*0.9503041f;
    X /= 0.95047f; Z /= 1.08883f;
    auto f = [](float t) {
        return (t > 0.008856f) ? std::cbrt(t) : (7.787f*t + 16.f/116.f);
    };
    float fx = f(X), fy = f(Y), fz = f(Z);
    return {116.f*fy - 16.f, 500.f*(fx - fy), 200.f*(fy - fz)};
}
static float labDistSq(uint32_t a, uint32_t b) {
    Lab la = rgbToLab(rgb24(a)), lb = rgbToLab(rgb24(b));
    float dL = la.L-lb.L, da = la.a-lb.a, db = la.b-lb.b;
    return dL*dL + da*da + db*db;
}

// ───────────────────────────────────────────────────────────────────────────
//  Stage 0 — Separable Gaussian pre-blur
//  Two-pass 1-D convolution.  sigma=0 → passthrough copy only.
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
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float r=0,g=0,b=0,a=0;
            for (int k = 0; k < ksize; ++k) {
                int sx = std::clamp(x+k-radius, 0, W-1);
                const uint8_t* p = src + (y*W+sx)*4;
                float w = kern[k];
                r += w*p[0]; g += w*p[1]; b += w*p[2]; a += w*p[3];
            }
            float* q = &tmp[(y*W+x)*4];
            q[0]=r; q[1]=g; q[2]=b; q[3]=a;
        }
    }
    // Vertical pass
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float r=0,g=0,b=0,a=0;
            for (int k = 0; k < ksize; ++k) {
                int sy = std::clamp(y+k-radius, 0, H-1);
                const float* q = &tmp[(sy*W+x)*4];
                float w = kern[k];
                r += w*q[0]; g += w*q[1]; b += w*q[2]; a += w*q[3];
            }
            uint8_t* d = dst.data() + (y*W+x)*4;
            d[0] = (uint8_t)std::clamp((int)(r+.5f),0,255);
            d[1] = (uint8_t)std::clamp((int)(g+.5f),0,255);
            d[2] = (uint8_t)std::clamp((int)(b+.5f),0,255);
            d[3] = (uint8_t)std::clamp((int)(a+.5f),0,255);
        }
    }
    return dst;
}

// ───────────────────────────────────────────────────────────────────────────
//  Stage 1 — Median-cut palette quantisation
//
//  Produces at most 2^color_precision representative colours that best
//  represent the actual pixel data.  Superior to bit-masking for photos
//  because it adapts to the image's actual colour distribution.
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

    // Collect unique colours + frequencies
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

    // Median-cut palette
    const int targetSz = std::min(1 << std::clamp(opt.color_precision,1,8),
                                  kMaxPaletteSize);
    std::vector<ColorEntry> entries;
    entries.reserve(freq.size());
    for (auto& [c,cnt]:freq) entries.push_back({rgb24(c),cnt});

    std::vector<uint32_t> palette = medianCutPalette(entries, targetSz);
    if (palette.empty()) return {};

    // Perceptual gradient merge via CIE-Lab Union-Find
    const int K = (int)palette.size();
    UnionFind uf(K);
    if (opt.gradient_step > 0.f) {
        const float thresh2 = opt.gradient_step * opt.gradient_step;
        for (int ii=0;ii<K;++ii)
            for (int jj=ii+1;jj<K;++jj)
                if (labDistSq(palette[ii],palette[jj]) <= thresh2)
                    uf.unite(ii,jj);
    }

    // Pixel-count-weighted average colour per group
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

    // Nearest-palette lookup with cache
    std::unordered_map<uint32_t,uint32_t> nearestCache;
    nearestCache.reserve(freq.size()*2);
    auto nearest = [&](uint32_t c) -> uint32_t {
        auto it = nearestCache.find(c);
        if (it != nearestCache.end()) return it->second;
        float bestD=1e30f; int bestI=0;
        for (int i=0;i<K;++i){
            float d=labDistSq(c,palette[i]);
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

    // Collect used canonical colours, sort darkest-first
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

// ───────────────────────────────────────────────────────────────────────────
//  Stage 5 — Moore boundary trace (Jacob's stopping criterion)
//  Pixel centres offset by +0.5 for sub-pixel accuracy.
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

// ───────────────────────────────────────────────────────────────────────────
//  Stage 6 — Ramer-Douglas-Peucker simplification
//  Iterative (stack-based) to avoid stack overflow on large paths.
// ───────────────────────────────────────────────────────────────────────────
static float ptSegDistSq(const Point& p, const Point& a, const Point& b) {
    float dx=b.x-a.x, dy=b.y-a.y;
    float lenSq=dx*dx+dy*dy;
    if (lenSq<1e-12f){float ex=p.x-a.x,ey=p.y-a.y;return ex*ex+ey*ey;}
    float t=std::clamp(((p.x-a.x)*dx+(p.y-a.y)*dy)/lenSq,0.f,1.f);
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
//  Stage 7 — Corner detection
//  A vertex is a hard corner when its turning angle < corner_threshold_deg.
//  Default 120°: only turns sharper than 60° become corners.
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
        float dot=std::clamp((ax*bx+ay*by)/(lA*lB),-1.f,1.f);
        c[i]=(std::acos(dot)<thresh_rad)?1:0;
    }
    return c;
}

// ───────────────────────────────────────────────────────────────────────────
//  Stage 8 — G1-continuous Catmull-Rom → cubic Bézier spline
//  Phantom control points at run boundaries are clamped to hard-corner
//  positions, ensuring G1 continuity across corner transitions.
// ───────────────────────────────────────────────────────────────────────────
static Segment catmullToBezier(
    const Point& p0, const Point& p1,
    const Point& p2, const Point& p3,
    float tension=0.5f)
{
    Point cp1={p1.x+tension*(p2.x-p0.x)/3.f,
               p1.y+tension*(p2.y-p0.y)/3.f};
    Point cp2={p2.x-tension*(p3.x-p1.x)/3.f,
               p2.y-tension*(p3.y-p1.y)/3.f};
    return Segment{true,cp1,cp2,p2};
}

static std::vector<Segment> buildSpline(
    const std::vector<Point>&   pts,
    const std::vector<uint8_t>& isCorner,
    float tension=0.5f)
{
    const int n=(int)pts.size();
    if (n==0) return {};
    std::vector<Segment> segs; segs.reserve(n);
    auto nxt=[&](int i){return (i+1)%n;};

    std::vector<int> corners;
    for (int i=0;i<n;++i) if (isCorner[i]) corners.push_back(i);
    if (corners.empty()) corners.push_back(0);

    const int nc=(int)corners.size();
    for (int ci=0;ci<nc;++ci){
        int from=corners[ci];
        int to=corners[(ci+1)%nc];
        std::vector<int> run; run.reserve(16);
        for (int k=from;;k=nxt(k)){
            run.push_back(k);
            if (k==to) break;
            if ((int)run.size()>n+2) break;
        }
        const int rn=(int)run.size();
        if (rn<2) continue;
        if (rn==2){
            segs.push_back({false,{},{},pts[run[1]]});
        } else {
            for (int ri=0;ri<rn-1;++ri){
                int ip0=run[std::max(ri-1,0)];
                int ip1=run[ri];
                int ip2=run[ri+1];
                int ip3=run[std::min(ri+2,rn-1)];
                segs.push_back(catmullToBezier(
                    pts[ip0],pts[ip1],pts[ip2],pts[ip3],tension));
            }
        }
    }
    return segs;
}

// ───────────────────────────────────────────────────────────────────────────
//  Stage 9 — Winding-order normalisation
//  SVG nonzero fill rule: outer paths CW, holes CCW (screen coords, Y-down).
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
//  Stage 10 — SVG output helpers
// ───────────────────────────────────────────────────────────────────────────

// Compact float: strips trailing zeros.  "142.500" → "142.5",  "337.0" → "337"
static void appendFloat(std::string& s, float v, int dp){
    char buf[32];
    int len=snprintf(buf,sizeof(buf),"%.*f",dp,(double)v);
    if (dp>0){
        while (len>1&&buf[len-1]=='0') --len;
        if (len>1&&buf[len-1]=='.') --len;
    }
    s.append(buf,len);
}

// Relative path commands — saves bytes when deltas are small.
// Emits 'h'/'v' optimisations for axis-aligned lines.
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

// 3-digit hex when all channel nibbles are equal (#ff0000 → #f00).
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

    // Stage 0: Gaussian pre-blur
    std::vector<uint8_t> blurred = gaussianBlur(pixels, width, height, options.blur_radius);
    const uint8_t* src = blurred.data();

    // Stages 1+2: median-cut quantise + Lab gradient merge
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

    // Stages 4–10: per-colour processing, darkest-first layer order
    for (uint32_t color : palette){
        if (!colorToComponents.count(color)) continue;

        std::fill(occ.begin(),occ.end(),0);
        for (int i=0;i<N;++i)
            if (pixelColor[i]==color) occ[i]=1;

        // Collect all paths for this colour
        struct PathRecord {
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
                    continue;
                }

                // Stage 5: boundary trace
                std::vector<Point> raw=traceBoundary(x,y,width,height,occ);
                clearComponent(idx,lbl,labelMap,occ,width,height);
                if ((int)raw.size()<3) continue;

                // Stage 6: RDP simplification
                std::vector<Point> simplified=rdpSimplify(raw,options.rdp_epsilon);
                if ((int)simplified.size()<3) continue;

                // Stage 7: corner detection
                std::vector<uint8_t> corners=
                    detectCorners(simplified,options.corner_threshold);

                // Stage 8: Bézier spline
                std::vector<Segment> segs=buildSpline(simplified,corners);
                if (segs.empty()) continue;

                paths.push_back({std::move(simplified),std::move(segs),false});
            }
        }
        if (paths.empty()) continue;

        // Stage 9: hole detection + winding order
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
            pr.segs=buildSpline(pr.pts,c2);
        }

        // Stage 10: emit SVG — group all paths by colour
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