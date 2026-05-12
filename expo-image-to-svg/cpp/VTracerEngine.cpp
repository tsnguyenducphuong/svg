// ═══════════════════════════════════════════════════════════════════════════
//  VTracerEngine.cpp
//  C++ port of visioncortex/vtracer  (https://github.com/visioncortex/vtracer)  
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

struct Point {
    float x, y;
};

// A path segment is either a line-to or a cubic Bézier curve.
struct Segment {
    bool  isCurve;   // false → line-to endpoint only; true → full cubic
    Point cp1;       // Bézier control point 1  (unused when isCurve==false)
    Point cp2;       // Bézier control point 2  (unused when isCurve==false)
    Point end;       // Line/curve endpoint
};

// ───────────────────────────────────────────────────────────────────────────
//  Moore Neighbourhood — 8-connected, clockwise order starting from North
//
//   7  0  1
//   6  *  2
//   5  4  3
// ───────────────────────────────────────────────────────────────────────────
static const int DX[8] = { 0,  1,  1,  1,  0, -1, -1, -1 };
static const int DY[8] = {-1, -1,  0,  1,  1,  1,  0, -1 };

// ───────────────────────────────────────────────────────────────────────────
//  Union-Find for gradient colour merging
// ───────────────────────────────────────────────────────────────────────────

struct UnionFind {
    std::vector<int> parent, rank_;

    explicit UnionFind(int n) : parent(n), rank_(n, 0) {
        for (int i = 0; i < n; ++i) parent[i] = i;
    }

    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]]; // path compression
            x = parent[x];
        }
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

// Euclidean RGB distance squared between two packed 0x00RRGGBB colours.
static float colorDistSq(uint32_t a, uint32_t b) {
    float dr = (float)((a >> 16) & 0xFF) - (float)((b >> 16) & 0xFF);
    float dg = (float)((a >>  8) & 0xFF) - (float)((b >>  8) & 0xFF);
    float db = (float)( a        & 0xFF) - (float)( b        & 0xFF);
    return dr*dr + dg*dg + db*db;
}

// Perceived luminance (BT.601) of a packed colour — used for layer ordering.
static float luminance(uint32_t c) {
    float r = (float)((c >> 16) & 0xFF);
    float g = (float)((c >>  8) & 0xFF);
    float b = (float)( c        & 0xFF);
    return 0.299f*r + 0.587f*g + 0.114f*b;
}

// ───────────────────────────────────────────────────────────────────────────
//  Stage 1 + 2 — Colour quantisation & gradient merging
//
//  Returns: palette (sorted darkest-first), and pixelColor[] mapping each
//           opaque pixel index → canonical palette colour (0x00RRGGBB).
//           Transparent pixels map to 0xFFFFFFFF (sentinel).
// ───────────────────────────────────────────────────────────────────────────

static std::vector<uint32_t> buildPaletteAndAssign(
    const uint8_t*          pixels,
    int                     width,
    int                     height,
    const Options&          opt,
    std::vector<uint32_t>&  pixelColor)   // out: one entry per pixel
{
    const int N = width * height;
    pixelColor.assign(N, 0xFFFFFFFFu);   // transparent sentinel

    // ── Step 1: quantise each opaque pixel ───────────────────────────────
    // Collect unique quantised colours and record which colour each pixel has.
    std::unordered_map<uint32_t, int> colorIndex; // colour → index in 'colors'
    std::vector<uint32_t> colors;                 // index → colour

    const uint8_t shift = (uint8_t)(8 - std::clamp(opt.color_precision, 1, 8));
    const uint8_t mask  = (uint8_t)(0xFF << shift);

    for (int i = 0; i < N; ++i) {
        const uint8_t* p = pixels + i * 4;
        if (p[3] < 128) continue;   // transparent — leave sentinel

        uint32_t qc;
        if (opt.color_mode == ColorMode::BlackAndWhite) {
            float lum = 0.299f*(float)p[0] + 0.587f*(float)p[1] + 0.114f*(float)p[2];
            qc = (lum >= 128.f) ? 0x00FFFFFFu : 0x00000000u;
        } else {
            uint8_t r = p[0] & mask;
            uint8_t g = p[1] & mask;
            uint8_t b = p[2] & mask;
            qc = ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
        }

        auto [it, inserted] = colorIndex.emplace(qc, (int)colors.size());
        if (inserted) colors.push_back(qc);
        pixelColor[i] = qc;
    }

    if (colors.empty()) return {};

    // ── Step 2: gradient merging via Union-Find ──────────────────────────
    // For every pair of palette entries within gradient_step, merge them.
    // We only compare neighbours in the sorted palette for efficiency (O(k²)
    // where k = palette size, which is small after quantisation).
    const int K = (int)colors.size();
    UnionFind uf(K);

    if (opt.gradient_step > 0.f) {
        const float thresh2 = opt.gradient_step * opt.gradient_step;
        // Sort palette by luminance so nearby entries cluster together.
        std::vector<int> order(K);
        for (int i = 0; i < K; ++i) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&](int a, int b){ return luminance(colors[a]) < luminance(colors[b]); });

        // Compare every pair — palette is small (≤ 256 after 6-bit quant).
        for (int ii = 0; ii < K; ++ii)
            for (int jj = ii+1; jj < K; ++jj)
                if (colorDistSq(colors[order[ii]], colors[order[jj]]) <= thresh2)
                    uf.unite(order[ii], order[jj]);
    }

    // Build canonical colour map: index → root colour (darkest of its group)
    // "Darkest" = lowest luminance, matching VTracer layer-stacking convention.
    std::unordered_map<int,uint32_t> rootColor;
    for (int i = 0; i < K; ++i) {
        int root = uf.find(i);
        auto it = rootColor.find(root);
        if (it == rootColor.end() || luminance(colors[i]) < luminance(it->second))
            rootColor[root] = colors[i];
    }

    // Re-map every pixel to its canonical colour.
    for (int i = 0; i < N; ++i) {
        if (pixelColor[i] == 0xFFFFFFFFu) continue;
        uint32_t qc = pixelColor[i];
        int      idx = colorIndex[qc];
        pixelColor[i] = rootColor[uf.find(idx)];
    }

    // Collect unique canonical colours and sort darkest-first.
    std::unordered_map<uint32_t,bool> seen;
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
//
//  FIX-PERF-1: Single label map instead of one bool vector per colour.
//
//  Returns a vector<int> of size W×H.
//    ≥ 0  → component label for that pixel
//    -1   → transparent / unassigned
//
//  Also returns componentColor: label → canonical colour.
// ───────────────────────────────────────────────────────────────────────────

static std::vector<int> labelComponents(
    const std::vector<uint32_t>& pixelColor,
    int                          width,
    int                          height,
    std::vector<uint32_t>&       componentColor,   // out
    std::vector<int>&            componentSize)     // out
{
    const int N = width * height;
    std::vector<int> label(N, -1);
    componentColor.clear();
    componentSize.clear();

    // 4-connectivity offsets
    static const int off4x[4] = { 1, -1,  0,  0 };
    static const int off4y[4] = { 0,  0,  1, -1 };

    int nextLabel = 0;
    std::queue<int> q;

    for (int i = 0; i < N; ++i) {
        if (pixelColor[i] == 0xFFFFFFFFu) continue; // transparent
        if (label[i] != -1) continue;               // already labelled

        // BFS flood fill
        uint32_t myColor = pixelColor[i];
        int lbl = nextLabel++;
        componentColor.push_back(myColor);
        componentSize.push_back(0);

        label[i] = lbl;
        q.push(i);

        while (!q.empty()) {
            int cur = q.front(); q.pop();
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
                q.push(ni);
            }
        }
    }

    return label;
}

// ───────────────────────────────────────────────────────────────────────────
//  Stage 4 (speckle filter) + Stage 5 — boundary tracing
//
//  FIX-BUG-1 + BUG-2 + BUG-3:
//    • Full BFS clears the ENTIRE cluster from the occupancy map after tracing.
//    • Pixels are marked visited inside traceBoundary as the walker moves.
//    • Jacob's stopping criterion: stop when we re-enter start from same dir.
//
//  The occupancy map (occ) is a mutable uint8_t array — 1 = occupied, shared
//  across all calls so cleared pixels are never re-visited.
// ───────────────────────────────────────────────────────────────────────────

// Erase every pixel belonging to component `lbl` from occupancy map using BFS.
static void clearComponent(
    int startIdx, int lbl,
    const std::vector<int>& labelMap,
    std::vector<uint8_t>& occ,
    int width, int height)
{
    static const int off4x[4] = { 1, -1, 0, 0 };
    static const int off4y[4] = { 0, 0, 1, -1 };

    // ENHANCEMENT: Flat vector queue avoids constant heap allocations
    std::vector<int> q;
    q.reserve(width * height); 
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
            q.push_back(ni); // Push to back
        }
    }
}
// Moore Neighbourhood boundary walk with Jacob's stopping criterion.
//
// FIX-BUG-2: marks pixels visited (occ → boundary marker 2) as we walk.
// FIX-BUG-3: Jacob's criterion — stop on (pos == start) AND (dir == entryDir).
//
// Returns the raw integer-coordinate boundary polygon.
static std::vector<Point> traceBoundary(
    int startX, int startY,
    int width,  int height,
    std::vector<uint8_t>& occ)
{
    std::vector<Point> path;
    // Find first occupied neighbour to establish entry direction.
    int entryDir = -1;
    for (int d = 0; d < 8; ++d) {
        int nx = startX + DX[d];
        int ny = startY + DY[d];
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        if (occ[ny * width + nx]) { entryDir = d; break; }
    }
    // Isolated single pixel — just return it.
    if (entryDir == -1) {
        path.push_back({ (float)startX, (float)startY });
        return path;
    }

    int curX = startX, curY = startY;
    // For Jacob's criterion we need the direction we CAME FROM when we first
    // entered the start pixel from outside (i.e., the direction that brought
    // us to start the second time around).
    int firstEntryFromDir = -1;

    // fromDir is (entryDir + 5) % 8: the "left" of the first outgoing step.
    int fromDir = (entryDir + 5) % 8;

    bool firstStep = true;
    const int maxSteps = width * height * 2 + 8; // generous safety cap

    for (int step = 0; step < maxSteps; ++step) {
        path.push_back({ (float)curX, (float)curY });
        occ[curY * width + curX] = 2; // mark as boundary-visited

        bool found = false;
        for (int i = 0; i < 8; ++i) {
            int dir  = (fromDir + i) % 8;
            int nx   = curX + DX[dir];
            int ny   = curY + DY[dir];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            if (!occ[ny * width + nx]) continue;

            // Jacob's stopping criterion:
            // We stop when we return to the start pixel AND we entered it from
            // the same relative direction as the very first step.
            if (nx == startX && ny == startY) {
                if (firstStep) {
                    firstEntryFromDir = dir;
                    firstStep = false;
                } else if (dir == firstEntryFromDir) {
                    return path; // closed — done
                }
            } else {
                firstStep = false;
            }

            curX = nx; curY = ny;
            fromDir = (dir + 5) % 8;
            found = true;
            break;
        }

        if (!found) break; // dead-end (isolated or single-pixel appendage)
    }

    return path;
}

// ───────────────────────────────────────────────────────────────────────────
//  Stage 6 — Collinear-step collapse
//
//  VTracer's raw boundary walk emits one point per pixel step; most of those
//  on a straight horizontal/vertical/diagonal run are collinear and should be
//  collapsed to just the two endpoints.  This dramatically reduces the number
//  of vertices before corner detection, improving both quality and speed.
// ───────────────────────────────────────────────────────────────────────────

static std::vector<Point> collapseCollinear(const std::vector<Point>& raw) {
    const int n = (int)raw.size();
    if (n <= 2) return raw;

    std::vector<Point> out;
    out.push_back(raw[0]);

    for (int i = 1; i < n - 1; ++i) {
        // Cross-product of (prev→cur) × (cur→next).
        // Zero means collinear — skip the middle point.
        float ax = raw[i].x   - raw[i-1].x;
        float ay = raw[i].y   - raw[i-1].y;
        float bx = raw[i+1].x - raw[i].x;
        float by = raw[i+1].y - raw[i].y;
        if (ax * by != ay * bx) // not collinear
            out.push_back(raw[i]);
    }

    out.push_back(raw[n-1]);
    return out;
}

// ───────────────────────────────────────────────────────────────────────────
//  Stage 7 — Corner detection
//
//  FIX-BUG-4: uses interior angle in degrees (matching VTracer's semantics),
//  NOT an RDP epsilon.  A vertex is a "hard corner" when the turning angle
//  (= 180° − interior angle) exceeds (180° − corner_threshold).
//
//  We compute the angle between the incoming and outgoing vectors using atan2.
//  If the angle is BELOW corner_threshold degrees the vertex is a corner.
// ───────────────────────────────────────────────────────────────────────────

static std::vector<uint8_t> detectCorners(
    const std::vector<Point>& pts,
    float corner_threshold_deg)
{
    const int n = (int)pts.size();
    std::vector<uint8_t> isCorner(n, false);
    if (n < 3) {
        for (int k = 0; k < n; ++k) isCorner[k] = true;
        return isCorner;
    }

    const float thresh_rad = corner_threshold_deg * (float)M_PI / 180.f;

    for (int i = 0; i < n; ++i) {
        const Point& prev = pts[(i + n - 1) % n];
        const Point& cur  = pts[i];
        const Point& next = pts[(i + 1) % n];

        float ax = cur.x  - prev.x,  ay = cur.y  - prev.y;
        float bx = next.x - cur.x,   by = next.y - cur.y;

        float lenA = std::sqrt(ax*ax + ay*ay);
        float lenB = std::sqrt(bx*bx + by*by);
        if (lenA < 1e-6f || lenB < 1e-6f) { isCorner[i] = true; continue; }

        // Interior angle via dot product
        float dot   = (ax*bx + ay*by) / (lenA * lenB);
        dot         = std::clamp(dot, -1.f, 1.f);
        float angle = std::acos(dot);           // 0..π

        // angle < thresh  →  sharp turn  →  corner
        isCorner[i] = (angle < thresh_rad);
    }

    return isCorner;
}

// ───────────────────────────────────────────────────────────────────────────
//  Stage 8 — Cubic Bézier spline fitting
//
//  FIX-BUG-6: replaces polyline-only output with smooth curves.
//
//  Algorithm: Catmull-Rom → cubic Bézier conversion with tension α = 0.5.
//  Between two consecutive corners the segment is split into sub-runs; each
//  run is fit with Catmull-Rom tangents converted to Bézier control points.
//
//  Corner vertices are always emitted as hard line-to commands so the outline
//  faithfully preserves sharp features (logos, text, hard edges).
// ───────────────────────────────────────────────────────────────────────────

// Convert four Catmull-Rom spine points to one cubic Bézier segment.
// α = 0.5 (centripetal parametrisation — minimises cusps).
static Segment catmullToBezier(
    const Point& p0, const Point& p1,
    const Point& p2, const Point& p3)
{
    // Catmull-Rom tangent at p1 and p2 (tension 0.5 = half the chord)
    const float alpha = 0.5f;
    Point cp1 = { p1.x + alpha*(p2.x - p0.x)/3.f,
                  p1.y + alpha*(p2.y - p0.y)/3.f };
    Point cp2 = { p2.x - alpha*(p3.x - p1.x)/3.f,
                  p2.y - alpha*(p3.y - p1.y)/3.f };
    return Segment{ true, cp1, cp2, p2 };
}

// Build the final segment list for one closed polygon.
static std::vector<Segment> buildSpline(
    const std::vector<Point>& pts,
    const std::vector<uint8_t>&  isCorner)
{
    const int n = (int)pts.size();
    if (n == 0) return {};

    std::vector<Segment> segs;
    segs.reserve(n);

    // Walk from each corner to the next, fitting smooth curves in between.
    // If there are no corners at all, treat the whole polygon as one smooth loop.
    auto next = [&](int i){ return (i + 1) % n; };

    // Collect corner indices.
    std::vector<int> corners;
    for (int i = 0; i < n; ++i)
        if (isCorner[i]) corners.push_back(i);

    if (corners.empty()) {
        // Fully smooth loop — treat pt[0] as a phantom corner.
        corners.push_back(0);
    }

    const int nc = (int)corners.size();
    for (int ci = 0; ci < nc; ++ci) {
        int from = corners[ci];
        int to   = corners[(ci + 1) % nc];

        // Collect the run of points from `from` to `to` (inclusive).
        std::vector<int> run;
        for (int k = from; ; k = next(k)) {
            run.push_back(k);
            if (k == to) break;
        }

        const int rn = (int)run.size();
        if (rn < 2) continue;

        if (rn == 2) {
            // Only two points — emit a line.
            segs.push_back({ false, {}, {}, pts[run[1]] });
        } else {
            // Fit Catmull-Rom curve through the run.
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
// ───────────────────────────────────────────────────────────────────────────

// Append a float to `s` with exactly `dp` decimal places (no std::stream).
static void appendFloat(std::string& s, float v, int dp) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.*f", dp, (double)v);
    s += buf;
}

// Append one path segment (line or cubic Bézier) to the SVG `d` attribute.
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

    const int dp    = std::clamp(options.path_precision, 0, 6);
    const int N     = width * height;

    // ── Stage 1+2: quantise + gradient merge ────────────────────────────
    std::vector<uint32_t> pixelColor;
    std::vector<uint32_t> palette =
        buildPaletteAndAssign(pixels, width, height, options, pixelColor);

    if (palette.empty()) {
        // Fully transparent image — return empty SVG.
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

    // ── Stages 4–8: per-component trace, simplify, spline ───────────────
    // Occupancy map: 0 = empty, 1 = occupied, 2 = boundary-visited.
    // We process one palette colour at a time (darkest-first) so the SVG
    // layer order is correct.
    //
    // PERF: one shared occ array, reset per colour with a targeted fill.

    // Collect components per palette colour.
    std::unordered_map<uint32_t, std::vector<int>> colorToComponents;
    for (int lbl = 0; lbl < (int)componentColor.size(); ++lbl)
        colorToComponents[componentColor[lbl]].push_back(lbl);

    // SVG output — PERF-5: pre-allocated string, no stringstream.
    std::string svg;
    svg.reserve((size_t)N * 12); // rough estimate
    {
        char hdr[256];
        snprintf(hdr, sizeof(hdr),
                 "<svg xmlns='http://www.w3.org/2000/svg' "
                 "viewBox='0 0 %d %d'>", width, height);
        svg += hdr;
    }

    // One occ array reused across all components (reset lazily per colour).
    std::vector<uint8_t> occ(N, 0);

    for (uint32_t color : palette) {  // darkest-first (FEAT-6)
        auto it = colorToComponents.find(color);
        if (it == colorToComponents.end()) continue;

        // Build occupancy for this colour's pixels only.
        // We reset only the pixels that were set last iteration.
        std::fill(occ.begin(), occ.end(), 0);
        for (int i = 0; i < N; ++i)
            if (pixelColor[i] == color) occ[i] = 1;

        // Format the fill colour once.
        char colorHex[16];
        snprintf(colorHex, sizeof(colorHex), "#%06x", color);

        // Iterate over all start pixels in raster order.
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                if (occ[idx] != 1) continue;  // not an unvisited start pixel

                int lbl = labelMap[idx];

                // Stage 4: speckle filter — FIX-BUG-5
                if (componentSize[lbl] < options.filter_speckle) {
                    clearComponent(idx, lbl, labelMap, occ, width, height);
                    continue;
                }

                // Stage 5: boundary trace — FIX-BUG-1, BUG-2, BUG-3
                std::vector<Point> rawPath =
                    traceBoundary(x, y, width, height, occ);

                // BFS-clear the entire component — FIX-BUG-1
                clearComponent(idx, lbl, labelMap, occ, width, height);

                if (rawPath.size() < 3) continue;

                // Stage 6: collapse collinear steps — FEAT-3
                std::vector<Point> collapsed = collapseCollinear(rawPath);
                if (collapsed.size() < 3) continue;

                // Stage 7: corner detection — FIX-BUG-4
                std::vector<uint8_t> corners =
                    detectCorners(collapsed, options.corner_threshold);

                // Stage 8: Bézier spline fitting — FIX-BUG-6
                std::vector<Segment> segs = buildSpline(collapsed, corners);
                if (segs.empty()) continue;

                // Stage 9: emit SVG path
                std::string d;
                d.reserve(segs.size() * 20);
                d += 'M';
                appendFloat(d, collapsed[0].x, dp); d += ' ';
                appendFloat(d, collapsed[0].y, dp);

                for (const Segment& seg : segs)
                    appendSegment(d, seg, dp);

                d += 'Z';

                svg += "<path fill='";
                svg += colorHex;
                svg += "' d='";
                svg += d;
                svg += "'/>";
            }
        }
    }

    svg += "</svg>";
    return svg;
}

} // namespace vtracer
