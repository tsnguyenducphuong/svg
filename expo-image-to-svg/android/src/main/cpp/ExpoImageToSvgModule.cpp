#include <jni.h>
#include <jsi/jsi.h>
#include <string>
#include "VTracerEngine.hpp"

using namespace facebook;

namespace expo::imagetosvg {


// =============================================================================
//  Helpers
// =============================================================================

// Unwrap a JS Uint8Array value into a raw uint8_t pointer + byte length.
// Throws a jsi::JSError on any type mismatch.
static uint8_t* unwrapUint8Array(
    jsi::Runtime& rt,
    const jsi::Object& params,
    const char* key,
    size_t& outByteLength)
{
    if (!params.hasProperty(rt, key)) {
        throw jsi::JSError(rt,
            std::string("[ExpoImageToSvg] Missing required property: ") + key + ".");
    }

    auto typedArrayObj = params.getPropertyAsObject(rt, key);

    if (!typedArrayObj.hasProperty(rt, "buffer")) {
        throw jsi::JSError(rt,
            std::string("[ExpoImageToSvg] Property \"") + key +
            "\" is not a valid Uint8Array (missing .buffer).");
    }

    auto rawBufferVal = typedArrayObj.getProperty(rt, "buffer");
    if (!rawBufferVal.isObject() || !rawBufferVal.asObject(rt).isArrayBuffer(rt)) {
        throw jsi::JSError(rt,
            std::string("[ExpoImageToSvg] \"") + key + "\".buffer is not an ArrayBuffer.");
    }

    auto arrayBuffer = rawBufferVal.asObject(rt).getArrayBuffer(rt);
    outByteLength = arrayBuffer.size(rt);
    return arrayBuffer.data(rt);
}


// Read an optional float property. Returns defaultValue if absent or non-number.
static float getOptFloat(
    jsi::Runtime& rt,
    const jsi::Object& obj,
    const char* key,
    float defaultValue)
{
    if (obj.hasProperty(rt, key) && obj.getProperty(rt, key).isNumber())
        return (float)obj.getProperty(rt, key).asNumber();
    return defaultValue;
}


// Read an optional int property. Returns defaultValue if absent or non-number.
static int getOptInt(
    jsi::Runtime& rt,
    const jsi::Object& obj,
    const char* key,
    int defaultValue)
{
    if (obj.hasProperty(rt, key) && obj.getProperty(rt, key).isNumber())
        return (int)obj.getProperty(rt, key).asNumber();
    return defaultValue;
}


// Read an optional bool property. Returns defaultValue if absent or non-bool.
static bool getOptBool(
    jsi::Runtime& rt,
    const jsi::Object& obj,
    const char* key,
    bool defaultValue)
{
    if (obj.hasProperty(rt, key) && obj.getProperty(rt, key).isBool())
        return obj.getProperty(rt, key).asBool();
    return defaultValue;
}


// ---------------------------------------------------------------------------
//  Map a JS PassOptions object → vtracer::Options.
//  Used for each of pass1, pass2, pass3 inside vectorizeMultiPassJSI.
//
//  JS key               C++ field                Notes
//  ─────────────────    ──────────────────────── ─────────────────────────
//  precision         →  color_precision           int
//  segmentThreshold  →  segment_threshold         float
//  threshold         →  corner_threshold          float (180 - threshold)
//  filterSpeckle     →  filter_speckle            int
//  rdpEpsilon        →  rdp_epsilon               float
//  fitTolerance      →  fit_tolerance             float
//  pathPrecision     →  path_precision            int
//  gradientDetectThresh → gradient_detect_thresh  float
//  bilateralSigmaR   →  bilateral_sigma_r         float
//  colorMode         →  color_mode                enum
// ---------------------------------------------------------------------------
static vtracer::Options parsePassOptions(
    jsi::Runtime& rt,
    const jsi::Object& passObj,
    const vtracer::Options& defaults)
{
    vtracer::Options o = defaults;

    if (passObj.hasProperty(rt, "precision") &&
        passObj.getProperty(rt, "precision").isNumber())
        o.color_precision = (int)passObj.getProperty(rt, "precision").asNumber();

    if (passObj.hasProperty(rt, "segmentThreshold") &&
        passObj.getProperty(rt, "segmentThreshold").isNumber())
        o.segment_threshold = (float)passObj.getProperty(rt, "segmentThreshold").asNumber();

    if (passObj.hasProperty(rt, "threshold") &&
        passObj.getProperty(rt, "threshold").isNumber())
        o.corner_threshold = (float)(180.0 - passObj.getProperty(rt, "threshold").asNumber());

    if (passObj.hasProperty(rt, "filterSpeckle") &&
        passObj.getProperty(rt, "filterSpeckle").isNumber())
        o.filter_speckle = (int)passObj.getProperty(rt, "filterSpeckle").asNumber();

    if (passObj.hasProperty(rt, "rdpEpsilon") &&
        passObj.getProperty(rt, "rdpEpsilon").isNumber())
        o.rdp_epsilon = (float)passObj.getProperty(rt, "rdpEpsilon").asNumber();

    if (passObj.hasProperty(rt, "fitTolerance") &&
        passObj.getProperty(rt, "fitTolerance").isNumber())
        o.fit_tolerance = (float)passObj.getProperty(rt, "fitTolerance").asNumber();

    if (passObj.hasProperty(rt, "pathPrecision") &&
        passObj.getProperty(rt, "pathPrecision").isNumber())
        o.path_precision = (int)passObj.getProperty(rt, "pathPrecision").asNumber();

    if (passObj.hasProperty(rt, "gradientDetectThresh") &&
        passObj.getProperty(rt, "gradientDetectThresh").isNumber())
        o.gradient_detect_thresh = (float)passObj.getProperty(rt, "gradientDetectThresh").asNumber();

    if (passObj.hasProperty(rt, "bilateralSigmaR") &&
        passObj.getProperty(rt, "bilateralSigmaR").isNumber())
        o.bilateral_sigma_r = (float)passObj.getProperty(rt, "bilateralSigmaR").asNumber();

    if (passObj.hasProperty(rt, "colorMode") &&
        passObj.getProperty(rt, "colorMode").isString()) {
        std::string cm = passObj.getProperty(rt, "colorMode").asString(rt).utf8(rt);
        o.color_mode = (cm == "blackAndWhite")
            ? vtracer::ColorMode::BlackAndWhite
            : vtracer::ColorMode::Color;
    }

    return o;
}


// =============================================================================
//  JSI host function: nativeVectorize  (single-pass, unchanged)
// =============================================================================
jsi::Value vectorizeJSI(
    jsi::Runtime& rt,
    const jsi::Value& /*thisValue*/,
    const jsi::Value* args,
    size_t count)
{
    if (count < 1 || !args[0].isObject())
        throw jsi::JSError(rt, "[ExpoImageToSvg] Expected an options object.");

    auto params = args[0].asObject(rt);

    // ── Buffer ───────────────────────────────────────────────────────────────
    size_t byteLength = 0;
    uint8_t* pixels = unwrapUint8Array(rt, params, "buffer", byteLength);

    // ── Dimensions ───────────────────────────────────────────────────────────
    if (!params.hasProperty(rt, "width") || !params.getProperty(rt, "width").isNumber())
        throw jsi::JSError(rt, "[ExpoImageToSvg] Missing or invalid property: width.");
    if (!params.hasProperty(rt, "height") || !params.getProperty(rt, "height").isNumber())
        throw jsi::JSError(rt, "[ExpoImageToSvg] Missing or invalid property: height.");

    int width  = (int)params.getProperty(rt, "width").asNumber();
    int height = (int)params.getProperty(rt, "height").asNumber();

    if (width <= 0 || height <= 0)
        throw jsi::JSError(rt, "[ExpoImageToSvg] Invalid image dimensions.");

    if (byteLength < (size_t)(width * height * 4))
        throw jsi::JSError(rt, "[ExpoImageToSvg] Buffer is too small for the given dimensions.");

    // ── Options ──────────────────────────────────────────────────────────────
    vtracer::Options opts;

    if (params.hasProperty(rt, "precision") && params.getProperty(rt, "precision").isNumber())
        opts.color_precision = (int)params.getProperty(rt, "precision").asNumber();

    if (params.hasProperty(rt, "gradientDetectThresh") && params.getProperty(rt, "gradientDetectThresh").isNumber())
        opts.gradient_detect_thresh = (float)params.getProperty(rt, "gradientDetectThresh").asNumber();

    if (params.hasProperty(rt, "colorMode") && params.getProperty(rt, "colorMode").isString()) {
        std::string cm = params.getProperty(rt, "colorMode").asString(rt).utf8(rt);
        opts.color_mode = (cm == "blackAndWhite")
            ? vtracer::ColorMode::BlackAndWhite
            : vtracer::ColorMode::Color;
    }

    if (params.hasProperty(rt, "filterSpeckle") && params.getProperty(rt, "filterSpeckle").isNumber())
        opts.filter_speckle = (int)params.getProperty(rt, "filterSpeckle").asNumber();

    if (params.hasProperty(rt, "rdpEpsilon") && params.getProperty(rt, "rdpEpsilon").isNumber())
        opts.rdp_epsilon = (float)params.getProperty(rt, "rdpEpsilon").asNumber();

    // JS `threshold` is the intuitive sharpness angle; C++ stores (180 - threshold).
    if (params.hasProperty(rt, "threshold") && params.getProperty(rt, "threshold").isNumber())
        opts.corner_threshold = (float)(180.0 - params.getProperty(rt, "threshold").asNumber());

    if (params.hasProperty(rt, "fitTolerance") && params.getProperty(rt, "fitTolerance").isNumber())
        opts.fit_tolerance = (float)params.getProperty(rt, "fitTolerance").asNumber();

    if (params.hasProperty(rt, "blurRadius") && params.getProperty(rt, "blurRadius").isNumber())
        opts.blur_radius = (float)params.getProperty(rt, "blurRadius").asNumber();

    if (params.hasProperty(rt, "pathPrecision") && params.getProperty(rt, "pathPrecision").isNumber())
        opts.path_precision = (int)params.getProperty(rt, "pathPrecision").asNumber();

    if (params.hasProperty(rt, "bilateralSigmaR") && params.getProperty(rt, "bilateralSigmaR").isNumber())
        opts.bilateral_sigma_r = (float)params.getProperty(rt, "bilateralSigmaR").asNumber();

    if (params.hasProperty(rt, "gradientDetectThresh") && params.getProperty(rt, "gradientDetectThresh").isNumber())
        opts.gradient_detect_thresh = (float)params.getProperty(rt, "gradientDetectThresh").asNumber();

    if (params.hasProperty(rt, "segmentThreshold") && params.getProperty(rt, "segmentThreshold").isNumber())
        opts.segment_threshold = (float)params.getProperty(rt, "segmentThreshold").asNumber();

    // ── Execute ──────────────────────────────────────────────────────────────
    std::string svg = vtracer::vectorize(pixels, width, height, opts);
    return jsi::String::createFromUtf8(rt, svg);
}


// =============================================================================
//  JSI host function: nativeVectorizeMultiPass  (NEW)
//
//  JS options object layout (all buffer keys hold Uint8Arrays):
//
//   Required:
//     originalBuffer   Uint8Array  — source RGBA pixels
//     blurBuffer       Uint8Array  — bilateral/gaussian blurred RGBA
//     highPassBuffer   Uint8Array  — high-pass detail RGBA
//     maskBuffer       Uint8Array  — subject mask (R≥128 = fg) RGBA
//     edgeMapBuffer    Uint8Array  — Canny edge map (strength in R) RGBA
//     width            number
//     height           number
//
//   Optional (top-level compositing):
//     baseDilateRadius     number  → multiPassOpts.baseDilateRadius
//     highPassGroupOpacity number  → multiPassOpts.highPassGroupOpacity
//     edgeStrokeWidth      number  → multiPassOpts.edgeStrokeWidth
//     edgeMinLuminance     number  → multiPassOpts.edgeMinLuminance
//     enableSubPassAO      bool    → multiPassOpts.enableSubPassAO
//
//   Optional (per-pass sub-objects):
//     pass1            PassOptions object → multiPassOpts.pass1
//     pass2            PassOptions object → multiPassOpts.pass2
//     pass3            PassOptions object → multiPassOpts.pass3
// =============================================================================
jsi::Value vectorizeMultiPassJSI(
    jsi::Runtime& rt,
    const jsi::Value& /*thisValue*/,
    const jsi::Value* args,
    size_t count)
{
    if (count < 1 || !args[0].isObject())
        throw jsi::JSError(rt, "[ExpoImageToSvg] vectorizeMultiPass: Expected an options object.");

    auto params = args[0].asObject(rt);

    // ── Dimensions ───────────────────────────────────────────────────────────
    if (!params.hasProperty(rt, "width") || !params.getProperty(rt, "width").isNumber())
        throw jsi::JSError(rt, "[ExpoImageToSvg] vectorizeMultiPass: Missing or invalid property: width.");
    if (!params.hasProperty(rt, "height") || !params.getProperty(rt, "height").isNumber())
        throw jsi::JSError(rt, "[ExpoImageToSvg] vectorizeMultiPass: Missing or invalid property: height.");

    int width  = (int)params.getProperty(rt, "width").asNumber();
    int height = (int)params.getProperty(rt, "height").asNumber();

    if (width <= 0 || height <= 0)
        throw jsi::JSError(rt, "[ExpoImageToSvg] vectorizeMultiPass: Invalid image dimensions.");

    const size_t expectedBytes = (size_t)(width * height * 4);

    // ── Unwrap all five buffers ───────────────────────────────────────────────
    size_t bl0, bl1, bl2, bl3, bl4;
    uint8_t* originalPixels  = unwrapUint8Array(rt, params, "originalBuffer",  bl0);
    uint8_t* blurPixels      = unwrapUint8Array(rt, params, "blurBuffer",      bl1);
    uint8_t* highPassPixels  = unwrapUint8Array(rt, params, "highPassBuffer",  bl2);
    uint8_t* maskPixels      = unwrapUint8Array(rt, params, "maskBuffer",      bl3);
    uint8_t* edgeMapPixels   = unwrapUint8Array(rt, params, "edgeMapBuffer",   bl4);

    // Sanity-check all buffer sizes.
    const char* bufKeys[] = {
        "originalBuffer", "blurBuffer", "highPassBuffer",
        "maskBuffer",     "edgeMapBuffer"
    };
    size_t bufLens[] = { bl0, bl1, bl2, bl3, bl4 };
    for (int i = 0; i < 5; ++i) {
        if (bufLens[i] < expectedBytes) {
            throw jsi::JSError(rt,
                std::string("[ExpoImageToSvg] vectorizeMultiPass: \"") + bufKeys[i] +
                "\" is too small for the given dimensions.");
        }
    }

    // ── Build MultiPassOptions ────────────────────────────────────────────────
    vtracer::MultiPassOptions mpOpts; // constructors set per-pass defaults

    // Top-level compositing controls
    mpOpts.baseDilateRadius    = getOptFloat(rt, params, "baseDilateRadius",    mpOpts.baseDilateRadius);
    mpOpts.highPassGroupOpacity= getOptFloat(rt, params, "highPassGroupOpacity",mpOpts.highPassGroupOpacity);
    mpOpts.edgeStrokeWidth     = getOptFloat(rt, params, "edgeStrokeWidth",     mpOpts.edgeStrokeWidth);
    mpOpts.edgeMinLuminance    = getOptInt  (rt, params, "edgeMinLuminance",    mpOpts.edgeMinLuminance);
    mpOpts.enableSubPassAO     = getOptBool (rt, params, "enableSubPassAO",     mpOpts.enableSubPassAO);

    // Per-pass options: if a JS sub-object is supplied, merge its keys over
    // the already-defaulted C++ Options struct for that pass.
    if (params.hasProperty(rt, "pass1") && params.getProperty(rt, "pass1").isObject())
        mpOpts.pass1 = parsePassOptions(rt, params.getPropertyAsObject(rt, "pass1"), mpOpts.pass1);

    if (params.hasProperty(rt, "pass2") && params.getProperty(rt, "pass2").isObject())
        mpOpts.pass2 = parsePassOptions(rt, params.getPropertyAsObject(rt, "pass2"), mpOpts.pass2);

    if (params.hasProperty(rt, "pass3") && params.getProperty(rt, "pass3").isObject())
        mpOpts.pass3 = parsePassOptions(rt, params.getPropertyAsObject(rt, "pass3"), mpOpts.pass3);

    // ── Execute ──────────────────────────────────────────────────────────────
    std::string svg = vtracer::vectorizeMultiPass(
        originalPixels,
        blurPixels,
        highPassPixels,
        maskPixels,
        edgeMapPixels,
        width, height,
        mpOpts);

    return jsi::String::createFromUtf8(rt, svg);
}


} // namespace expo::imagetosvg


// =============================================================================
//  JNI Entry Point
//  Called from Kotlin/Java to install both native functions into the JS runtime.
// =============================================================================
extern "C"
JNIEXPORT void JNICALL
Java_expo_modules_imagetosvg_ExpoImageToSvgModule_installJSIBindings(
    JNIEnv* /*env*/,
    jobject /* thiz */,
    jlong jsiRuntimePtr)
{
    if (jsiRuntimePtr == 0) return;

    auto& runtime = *reinterpret_cast<facebook::jsi::Runtime*>(jsiRuntimePtr);

    // ── nativeVectorize (single-pass) ────────────────────────────────────────
    auto vectorizeFunc = facebook::jsi::Function::createFromHostFunction(
        runtime,
        facebook::jsi::PropNameID::forAscii(runtime, "nativeVectorize"),
        1,
        expo::imagetosvg::vectorizeJSI);
    runtime.global().setProperty(runtime, "nativeVectorize", std::move(vectorizeFunc));

    // ── nativeVectorizeMultiPass (frequency separation) ──────────────────────
    auto multiPassFunc = facebook::jsi::Function::createFromHostFunction(
        runtime,
        facebook::jsi::PropNameID::forAscii(runtime, "nativeVectorizeMultiPass"),
        1,
        expo::imagetosvg::vectorizeMultiPassJSI);
    runtime.global().setProperty(runtime, "nativeVectorizeMultiPass", std::move(multiPassFunc));
}