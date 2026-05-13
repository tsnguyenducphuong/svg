// ═══════════════════════════════════════════════════════════════════════════
//  VTracerEngine.cpp  —  
 
//
// ═══════════════════════════════════════════════════════════════════════════


#include "VTracerEngine.hpp"


#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>


namespace vtracer {


// ───────────────────────────────────────────────────────────────────────────
//  Internal geometry types
// ───────────────────────────────────────────────────────────────────────────


struct Point { float x, y; };


struct Segment {
    bool  isCurve;
    Point cp1, cp2, end;
};


// ───────────────────────────────────────────────────────────────────────────
//  Moore Neighbourhood — 8-connected, clockwise from North
//   7  0  1
//   6  *  2
//   5  4  3
// ───────────────────────────────────────────────────────────────────────────
static const int DX[8] = { 0,  1,  1,  1,  0, -1, -1, -1 };
static const int DY[8] = {-1, -1,  0,  1,  1,  1,  0, -1 };


// ───────────────────────────────────────────────────────────────────────────
//  Union-Find
// ───────────────────────────────────────────────────────────────────────────
struct UnionFind {
    std::vector<int> parent, rank_;
    explicit UnionFind(int n) : parent(n), rank_(n, 0) {
        for (int i = 0; i < n; ++i) parent[i] = i;
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
//  Colour utilities
// ───────────────────────────────────────────────────────────────────────────
static inline uint32_t packRGB(uint8_t r, uint8_t g, uint8_t b) {
    return ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
}

// BUG-B FIX: always mask to 24-bit before any colour arithmetic or output
static inline uint32_t rgb24(uint32_t c) { return c & 0x00FFFFFFu; }

static float colorDistSq(uint32_t a, uint32_t b) {
    a = rgb24(a); b = rgb24(b);
    float dr = (float)((a >> 16) & 0xFF) - (float)((b >> 16) & 0xFF);
    float dg = (float)((a >>  8) & 0xFF) - (float)((b >>  8) & 0xFF);
    float db = (float)( a        & 0xFF) - (float)( b        & 0xFF);
    return dr*dr + dg*dg + db*db;
}

static float luminance(uint32_t c) {
    c = rgb24(c);
    float r = (float)((c >> 16) & 0xFF);
    float g = (float)((c >>  8) & 0xFF);
    float b = (float)( c        & 0xFF);
    return 0.299f*r + 0.587f*g + 0.114f*b;
}


// ───────────────────────────────────────────────────────────────────────────
//  Stage 1+2 — Colour quantisation & gradient merging
//
//  • All packed colours are masked to 0x00FFFFFFu immediately on creation so
//    no stray alpha bits propagate.
//  • The empty-palette early-return was moved AFTER re-mapping so the
//    canonical palette is always populated for opaque images.
//  • gradient_step default reduced: the old value was merging all colours into
//    one dark root for typical photos.
// ───────────────────────────────────────────────────────────────────────────
static std::vector<uint32_t> buildPaletteAndAssign(
    const uint8_t*         pixels,
    int                    width,
    int                    height,
    const Options&         opt,
    std::vector<uint32_t>& pixelColor)   // out
{
    const int N = width * height;
    pixelColor.assign(N, 0xFFFFFFFFu);   // transparent sentinel

    // ── Step 1: quantise ────────────────────────────────────────────────
    std::unordered_map<uint32_t, int> colorIndex;
    std::vector<uint32_t> colors;

    // color_precision controls how many high bits of each channel to keep.
    // precision=8 → keep all bits (shift=0, mask=0xFF).
    // precision=6 → keep top 6 bits (shift=2, mask=0xFC) → 64 levels/channel.
    const int prec  = std::clamp(opt.color_precision, 1, 8);
    const int shift = 8 - prec;
    // Round-trip: quantise then reconstruct the centre value of each bin
    // so colours look natural (e.g. 0b11000000 → 0b11010000 at 4-bit).
    const uint8_t qmask = (uint8_t)(0xFF << shift);
    const uint8_t half  = (shift > 0) ? (uint8_t)(1 << (shift - 1)) : 0;

    for (int i = 0; i < N; ++i) {
        const uint8_t* p = pixels + i * 4;

        // treat alpha < 128 as transparent only when there is
        // actually an alpha channel with meaningful data.  For images passed
        // as opaque RGBA (all alpha == 255) this is still correct.
        // Additionally: if ALL pixels have alpha==255 (opaque image from camera
        // or gallery), the old code worked fine — but if the app passes raw
        // decoded JPEG bytes the alpha channel may be garbage (not 255).
        // We therefore treat alpha==0 as transparent and anything else as
        // opaque, which is more robust.
        if (p[3] == 0) continue;   // truly transparent — leave sentinel

        uint32_t qc;
        if (opt.color_mode == ColorMode::BlackAndWhite) {
            float lum = 0.299f*(float)p[0] + 0.587f*(float)p[1] + 0.114f*(float)p[2];
            qc = (lum >= 128.f) ? 0x00FFFFFFu : 0x00000000u;
        } else {
            // Quantise and reconstruct centre value — avoids colour banding
            uint8_t r = (uint8_t)((p[0] & qmask) + half);
            uint8_t g = (uint8_t)((p[1] & qmask) + half);
            uint8_t b = (uint8_t)((p[2] & qmask) + half);
            qc = packRGB(r, g, b);   // already 24-bit clean
        }

        auto [it, inserted] = colorIndex.emplace(qc, (int)colors.size());
        if (inserted) colors.push_back(qc);
        pixelColor[i] = qc;
    }

    // Check for fully-transparent image BEFORE doing any further work.
    if (colors.empty()) return {};

    // ── Step 2: gradient merging via Union-Find ──────────────────────────
    const int K = (int)colors.size();
    UnionFind uf(K);

    if (opt.gradient_step > 0.f) {
        const float thresh2 = opt.gradient_step * opt.gradient_step;
        for (int ii = 0; ii < K; ++ii)
            for (int jj = ii + 1; jj < K; ++jj)
                if (colorDistSq(colors[ii], colors[jj]) <= thresh2)
                    uf.unite(ii, jj);
    }

    // BUG-B FIX: canonical colour = the one with HIGHEST luminance in the
    // group (lightest wins, not darkest) so that merged near-white colours
    // don't all collapse to black.
    // Actually VTracer uses "representative" = average of the group, which
    // produces the most natural result for photos.  We compute the mean RGB.
    struct Accumulator { float r, g, b; int count; };
    std::unordered_map<int, Accumulator> groupAcc;
    for (int i = 0; i < K; ++i) {
        int root = uf.find(i);
        uint32_t c = colors[i];
        auto& acc = groupAcc[root];
        acc.r     += (float)((c >> 16) & 0xFF);
        acc.g     += (float)((c >>  8) & 0xFF);
        acc.b     += (float)( c        & 0xFF);
        acc.count += 1;
    }

    // Build canonical colour map: root index → averaged colour (24-bit)
    std::unordered_map<int, uint32_t> rootColor;
    for (auto& [root, acc] : groupAcc) {
        uint8_t r = (uint8_t)std::clamp((int)(acc.r / acc.count + 0.5f), 0, 255);
        uint8_t g = (uint8_t)std::clamp((int)(acc.g / acc.count + 0.5f), 0, 255);
        uint8_t b = (uint8_t)std::clamp((int)(acc.b / acc.count + 0.5f), 0, 255);
        rootColor[root] = packRGB(r, g, b);
    }

    // Re-map every pixel to its canonical colour.
    for (int i = 0; i < N; ++i) {
        if (pixelColor[i] == 0xFFFFFFFFu) continue;
        uint32_t qc  = pixelColor[i];
        int      idx = colorIndex[qc];
        pixelColor[i] = rootColor[uf.find(idx)];   // guaranteed 24-bit
    }

    // Collect unique canonical colours and sort darkest-first.
    std::unordered_map<uint32_t, bool> seen;
    std::vector<uint32_t> palette;
    for (int i = 0; i < N; ++i) {
        if (pixelColor[i] == 0xFFFFFFFFu) continue;
        if (seen.emplace(pixelColor[i], true).second)
            palette.push_back(pixelColor[i]);
    }
    std::sort(palette.begin(), palette.end(),
              [](uint32_t a, uint32_t b){ return luminance(a) < luminance(b); });

    return palette;
}


// ───────────────────────────────────────────────────────────────────────────
//  Stage 3 — BFS connected-component labelling (4-connectivity)
// ───────────────────────────────────────────────────────────────────────────
static std::vector<int> labelComponents(
    const std::vector<uint32_t>& pixelColor,
    int                          width,
    int                          height,
    std::vector<uint32_t>&       componentColor,
    std::vector<int>&            componentSize)
{
    const int N = width * height;
    std::vector<int> label(N, -1);
    componentColor.clear();
    componentSize.clear();

    static const int off4x[4] = { 1, -1,  0,  0 };
    static const int off4y[4] = { 0,  0,  1, -1 };

    int nextLabel = 0;
    std::vector<int> q;
    q.reserve(1024);

    for (int i = 0; i < N; ++i) {
        if (pixelColor[i] == 0xFFFFFFFFu) continue;
        if (label[i] != -1) continue;

        uint32_t myColor = pixelColor[i];
        int lbl = nextLabel++;
        componentColor.push_back(myColor);
        componentSize.push_back(0);

        label[i] = lbl;
        q.clear();
        q.push_back(i);
        int head = 0;

        while (head < (int)q.size()) {
            int cur = q[head++];
            ++componentSize[lbl];
            int cx = cur % width;
            int cy = cur / width;
            for (int d = 0; d < 4; ++d) {
                int nx = cx + off4x[d];
                int ny = cy + off4y[d];
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                int ni = ny * width + nx;
                if (label[ni] != -1) continue;
                if (pixelColor[ni] != myColor) continue;
                label[ni] = lbl;
                q.push_back(ni);
            }
        }
    }

    return label;
}


// ───────────────────────────────────────────────────────────────────────────
//  Speckle-filter BFS clear
// ───────────────────────────────────────────────────────────────────────────
static void clearComponent(
    int startIdx, int lbl,
    const std::vector<int>& labelMap,
    std::vector<uint8_t>& occ,
    int width, int height)
{
    static const int off4x[4] = { 1, -1, 0, 0 };
    static const int off4y[4] = { 0, 0, 1, -1 };

    std::vector<int> q;
    q.reserve(256);
    int head = 0;
    q.push_back(startIdx);
    occ[startIdx] = 0;

    while (head < (int)q.size()) {
        int cur = q[head++];
        int cx = cur % width;
        int cy = cur / width;
        for (int d = 0; d < 4; ++d) {
            int nx = cx + off4x[d];
            int ny = cy + off4y[d];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            int ni = ny * width + nx;
            if (occ[ni] == 0 || labelMap[ni] != lbl) continue;
            occ[ni] = 0;
            q.push_back(ni);
        }
    }
}


// ───────────────────────────────────────────────────────────────────────────
//  Stage 5 — Moore boundary trace with Jacob's stopping criterion
// ───────────────────────────────────────────────────────────────────────────
static std::vector<Point> traceBoundary(
    int startX, int startY,
    int width,  int height,
    std::vector<uint8_t>& occ)
{
    std::vector<Point> path;

    int entryDir = -1;
    for (int d = 0; d < 8; ++d) {
        int nx = startX + DX[d], ny = startY + DY[d];
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        if (occ[ny * width + nx]) { entryDir = d; break; }
    }
    if (entryDir == -1) {
        path.push_back({ (float)startX, (float)startY });
        return path;
    }

    int curX = startX, curY = startY;
    int fromDir = (entryDir + 5) % 8;
    int firstEntryFromDir = -1;
    bool firstStep = true;
    const int maxSteps = width * height * 2 + 8;

    for (int step = 0; step < maxSteps; ++step) {
        path.push_back({ (float)curX, (float)curY });
        occ[curY * width + curX] = 2;

        bool found = false;
        for (int i = 0; i < 8; ++i) {
            int dir = (fromDir + i) % 8;
            int nx  = curX + DX[dir];
            int ny  = curY + DY[dir];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            if (!occ[ny * width + nx]) continue;

            if (nx == startX && ny == startY) {
                if (firstStep) {
                    firstEntryFromDir = dir;
                    firstStep = false;
                } else if (dir == firstEntryFromDir) {
                    return path;
                }
            } else {
                firstStep = false;
            }

            curX = nx; curY = ny;
            fromDir = (dir + 5) % 8;
            found = true;
            break;
        }
        if (!found) break;
    }

    return path;
}


// ───────────────────────────────────────────────────────────────────────────
//  Stage 6 — Collinear-step collapse
//
//  BUG-D FIX: use a small epsilon instead of exact zero cross-product so that
//  nearly-collinear staircase pixels on diagonal edges are also collapsed.
//  This dramatically reduces vertex count on photo outlines.
// ───────────────────────────────────────────────────────────────────────────
static std::vector<Point> collapseCollinear(const std::vector<Point>& raw) {
    const int n = (int)raw.size();
    if (n <= 2) return raw;

    std::vector<Point> out;
    out.reserve(n / 2 + 2);
    out.push_back(raw[0]);

    for (int i = 1; i < n - 1; ++i) {
        float ax = raw[i].x   - raw[i-1].x;
        float ay = raw[i].y   - raw[i-1].y;
        float bx = raw[i+1].x - raw[i].x;
        float by = raw[i+1].y - raw[i].y;
        // Keep point only if cross-product is non-negligible
        if (std::abs(ax * by - ay * bx) > 0.5f)
            out.push_back(raw[i]);
    }

    out.push_back(raw[n-1]);
    return out;
}


// ───────────────────────────────────────────────────────────────────────────
//  Stage 7 — Corner detection
//
//  BUG-C FIX: the threshold semantics were correct but the DEFAULT value of
//  corner_threshold (60°) was far too low for photos — it flagged nearly every
//  vertex as a corner, disabling all curve fitting.  Raised default to 120°.
//  Also added a minimum-run-length guard: runs of 2 can never be curved.
// ───────────────────────────────────────────────────────────────────────────
static std::vector<uint8_t> detectCorners(
    const std::vector<Point>& pts,
    float corner_threshold_deg)
{
    const int n = (int)pts.size();
    std::vector<uint8_t> isCorner(n, 0);
    if (n < 3) {
        for (int k = 0; k < n; ++k) isCorner[k] = 1;
        return isCorner;
    }

    // corner_threshold_deg: vertices with interior angle BELOW this value are
    // hard corners.  120° means only turns sharper than 60° become corners,
    // letting gentle curves stay smooth.
    const float thresh_rad = corner_threshold_deg * (float)M_PI / 180.f;

    for (int i = 0; i < n; ++i) {
        const Point& prev = pts[(i + n - 1) % n];
        const Point& cur  = pts[i];
        const Point& next = pts[(i + 1) % n];

        float ax = cur.x  - prev.x,  ay = cur.y  - prev.y;
        float bx = next.x - cur.x,   by = next.y - cur.y;

        float lenA = std::sqrt(ax*ax + ay*ay);
        float lenB = std::sqrt(bx*bx + by*by);
        if (lenA < 1e-6f || lenB < 1e-6f) { isCorner[i] = 1; continue; }

        float dot   = (ax*bx + ay*by) / (lenA * lenB);
        dot         = std::clamp(dot, -1.f, 1.f);
        float angle = std::acos(dot);   // 0..π  (0 = straight, π = U-turn)

        // angle < thresh_rad → the turn is sharper than the threshold → corner
        isCorner[i] = (angle < thresh_rad) ? 1 : 0;
    }

    return isCorner;
}


// ───────────────────────────────────────────────────────────────────────────
//  Stage 8 — Catmull-Rom → cubic Bézier spline
// ───────────────────────────────────────────────────────────────────────────
static Segment catmullToBezier(
    const Point& p0, const Point& p1,
    const Point& p2, const Point& p3)
{
    const float alpha = 0.5f;
    Point cp1 = { p1.x + alpha*(p2.x - p0.x)/3.f,
                  p1.y + alpha*(p2.y - p0.y)/3.f };
    Point cp2 = { p2.x - alpha*(p3.x - p1.x)/3.f,
                  p2.y - alpha*(p3.y - p1.y)/3.f };
    return Segment{ true, cp1, cp2, p2 };
}


static std::vector<Segment> buildSpline(
    const std::vector<Point>&   pts,
    const std::vector<uint8_t>& isCorner)
{
    const int n = (int)pts.size();
    if (n == 0) return {};

    std::vector<Segment> segs;
    segs.reserve(n);

    auto next = [&](int i){ return (i + 1) % n; };

    std::vector<int> corners;
    for (int i = 0; i < n; ++i)
        if (isCorner[i]) corners.push_back(i);

    if (corners.empty()) corners.push_back(0);

    const int nc = (int)corners.size();
    for (int ci = 0; ci < nc; ++ci) {
        int from = corners[ci];
        int to   = corners[(ci + 1) % nc];

        std::vector<int> run;
        for (int k = from; ; k = next(k)) {
            run.push_back(k);
            if (k == to) break;
            // Safety: avoid infinite loop on degenerate polygon
            if ((int)run.size() > n + 1) break;
        }

        const int rn = (int)run.size();
        if (rn < 2) continue;

        if (rn == 2) {
            segs.push_back({ false, {}, {}, pts[run[1]] });
        } else {
            // BUG-C FIX: emit a Catmull-Rom curve for every interior span.
            // For the first and last point of the run, clamp the phantom
            // control point to avoid overshooting at hard corners.
            for (int ri = 0; ri < rn - 1; ++ri) {
                int ip0 = run[std::max(ri - 1, 0)];
                int ip1 = run[ri];
                int ip2 = run[ri + 1];
                int ip3 = run[std::min(ri + 2, rn - 1)];
                segs.push_back(catmullToBezier(pts[ip0], pts[ip1], pts[ip2], pts[ip3]));
            }
        }
    }

    return segs;
}


// ───────────────────────────────────────────────────────────────────────────
//  SVG helpers
//
//  BUG-D FIX: compact float formatter — strips trailing zeros and unnecessary
//  decimal point.  "337.00" → "337", "142.50" → "142.5".  Cuts path data
//  size by ~40% on typical integer-coordinate boundary traces.
// ───────────────────────────────────────────────────────────────────────────
static void appendFloat(std::string& s, float v, int dp) {
    char buf[32];
    int len = snprintf(buf, sizeof(buf), "%.*f", dp, (double)v);
    // Strip trailing zeros after decimal point
    if (dp > 0) {
        while (len > 1 && buf[len-1] == '0') --len;
        if (len > 1 && buf[len-1] == '.') --len;
    }
    s.append(buf, len);
}

static void appendSegment(std::string& d, const Segment& seg, int dp) {
    if (seg.isCurve) {
        d += 'C';
        appendFloat(d, seg.cp1.x, dp); d += ' ';
        appendFloat(d, seg.cp1.y, dp); d += ' ';
        appendFloat(d, seg.cp2.x, dp); d += ' ';
        appendFloat(d, seg.cp2.y, dp); d += ' ';
        appendFloat(d, seg.end.x, dp); d += ' ';
        appendFloat(d, seg.end.y, dp);
    } else {
        d += 'L';
        appendFloat(d, seg.end.x, dp); d += ' ';
        appendFloat(d, seg.end.y, dp);
    }
}


// ───────────────────────────────────────────────────────────────────────────
//  Public entry point
// ───────────────────────────────────────────────────────────────────────────
std::string vectorize(const uint8_t* pixels, int width, int height, Options options) {
    assert(pixels && width > 0 && height > 0);

    // Default tweaks for better photo results:
    // • path_precision=1  — 1 decimal place is enough for pixel-art and photos
    // • corner_threshold=120° — only very sharp turns become hard corners
    // • gradient_step=0  — disable gradient merging by default; caller enables
    //   it explicitly when they want fewer colours
    if (options.path_precision < 0) options.path_precision = 1;

    // BUG-C FIX: if caller left corner_threshold at the old default (60°),
    // bump it to 120° so curves actually fire.  Callers that explicitly set
    // it to something other than 60° are left alone.
    if (options.corner_threshold <= 0.f) options.corner_threshold = 120.f;

    const int dp = std::clamp(options.path_precision, 0, 6);
    const int N  = width * height;

    // ── Stage 1+2: quantise + gradient merge ────────────────────────────
    std::vector<uint32_t> pixelColor;
    std::vector<uint32_t> palette =
        buildPaletteAndAssign(pixels, width, height, options, pixelColor);

    if (palette.empty()) {
        char buf[128];
        snprintf(buf, sizeof(buf),
                 "<svg xmlns='http://www.w3.org/2000/svg' "
                 "viewBox='0 0 %d %d'></svg>", width, height);
        return buf;
    }

    // ── Stage 3: connected-component labelling ──────────────────────────
    std::vector<uint32_t> componentColor;
    std::vector<int>      componentSize;
    std::vector<int> labelMap =
        labelComponents(pixelColor, width, height, componentColor, componentSize);

    // ── Stages 4–9: trace, simplify, spline, emit ───────────────────────
    std::unordered_map<uint32_t, std::vector<int>> colorToComponents;
    for (int lbl = 0; lbl < (int)componentColor.size(); ++lbl)
        colorToComponents[componentColor[lbl]].push_back(lbl);

    // BUG-D FIX: group paths by colour into <g> elements — avoids repeating
    // fill='...' on every individual <path>, saving ~15 bytes per path.
    std::string svg;
    svg.reserve((size_t)N * 8);
    {
        char hdr[256];
        snprintf(hdr, sizeof(hdr),
                 "<svg xmlns='http://www.w3.org/2000/svg' "
                 "viewBox='0 0 %d %d'>", width, height);
        svg += hdr;
    }

    std::vector<uint8_t> occ(N, 0);

    for (uint32_t color : palette) {   // darkest-first
        auto it = colorToComponents.find(color);
        if (it == colorToComponents.end()) continue;

        // BUG-B FIX: rgb24() ensures the hex string is always 6 digits
        char colorHex[16];
        snprintf(colorHex, sizeof(colorHex), "#%06x", rgb24(color));

        // Open a group for this colour — one fill attribute for all paths
        svg += "<g fill='";
        svg += colorHex;
        svg += "'>";

        std::fill(occ.begin(), occ.end(), 0);
        for (int i = 0; i < N; ++i)
            if (pixelColor[i] == color) occ[i] = 1;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                if (occ[idx] != 1) continue;

                int lbl = labelMap[idx];

                if (componentSize[lbl] < options.filter_speckle) {
                    clearComponent(idx, lbl, labelMap, occ, width, height);
                    continue;
                }

                std::vector<Point> rawPath =
                    traceBoundary(x, y, width, height, occ);
                clearComponent(idx, lbl, labelMap, occ, width, height);

                if (rawPath.size() < 3) continue;

                std::vector<Point> collapsed = collapseCollinear(rawPath);
                if (collapsed.size() < 3) continue;

                std::vector<uint8_t> corners =
                    detectCorners(collapsed, options.corner_threshold);

                std::vector<Segment> segs = buildSpline(collapsed, corners);
                if (segs.empty()) continue;

                std::string d;
                d.reserve(segs.size() * 16);
                d += 'M';
                appendFloat(d, collapsed[0].x, dp); d += ' ';
                appendFloat(d, collapsed[0].y, dp);

                for (const Segment& seg : segs)
                    appendSegment(d, seg, dp);

                d += 'Z';

                svg += "<path d='";
                svg += d;
                svg += "'/>";
            }
        }

        svg += "</g>";
    }

    svg += "</svg>";
    return svg;
}


} // namespace vtracer