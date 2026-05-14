#include <jni.h>
#include <jsi/jsi.h>
#include <string>
#include "VTracerEngine.hpp"

using namespace facebook;

namespace expo::imagetosvg {

jsi::Value vectorizeJSI(
    jsi::Runtime& rt,
    const jsi::Value& thisValue,
    const jsi::Value* args,
    size_t count
) {
    // --- Argument validation ---
    if (count < 1 || !args[0].isObject()) {
        throw jsi::JSError(rt, "[ExpoImageToSvg] Expected an options object as the first argument.");
    }

    auto params = args[0].asObject(rt); 
    
    // -------------------------------------------------------------------------

    // Step 1 – grab the Uint8Array passed under the "buffer" key
    if (!params.hasProperty(rt, "buffer")) {
        throw jsi::JSError(rt, "[ExpoImageToSvg] Missing required property: buffer.");
    }
    auto typedArrayObj = params.getPropertyAsObject(rt, "buffer");

    // Step 2 – unwrap the underlying ArrayBuffer from the Uint8Array
    if (!typedArrayObj.hasProperty(rt, "buffer")) {
        throw jsi::JSError(rt, "[ExpoImageToSvg] Provided buffer is not a valid Uint8Array (missing .buffer).");
    }
    auto rawBufferVal = typedArrayObj.getProperty(rt, "buffer");

    // Step 3 – safety check before casting
    if (!rawBufferVal.isObject() || !rawBufferVal.asObject(rt).isArrayBuffer(rt)) {
        throw jsi::JSError(rt, "[ExpoImageToSvg] buffer.buffer is not an ArrayBuffer.");
    }

    auto arrayBuffer = rawBufferVal.asObject(rt).getArrayBuffer(rt);
    uint8_t* pixels  = arrayBuffer.data(rt);
    size_t byteLength = arrayBuffer.size(rt);

    // -------------------------------------------------------------------------
    // Extract & validate dimensions
    // -------------------------------------------------------------------------
    if (!params.hasProperty(rt, "width") || !params.getProperty(rt, "width").isNumber()) {
        throw jsi::JSError(rt, "[ExpoImageToSvg] Missing or invalid property: width.");
    }
    if (!params.hasProperty(rt, "height") || !params.getProperty(rt, "height").isNumber()) {
        throw jsi::JSError(rt, "[ExpoImageToSvg] Missing or invalid property: height.");
    }

    int width  = (int)params.getProperty(rt, "width").asNumber();
    int height = (int)params.getProperty(rt, "height").asNumber();

    if (width <= 0 || height <= 0) {
        throw jsi::JSError(rt, "[ExpoImageToSvg] Invalid image dimensions.");
    }

    // Sanity-check: buffer must be large enough for the declared dimensions (RGBA = 4 bytes/px).
    if (byteLength < (size_t)(width * height * 4)) {
        throw jsi::JSError(rt, "[ExpoImageToSvg] Buffer is too small for the given dimensions.");
    }

    // -------------------------------------------------------------------------
    // Map JS options -> vtracer::Options
    //
    // JS key            C++ field              Type     Notes
    // ──────────────    ─────────────────────  ───────  ─────────────────────────────────
    // precision      -> color_precision         int
    // gradientStep   -> gradient_step           float
    // colorMode      -> color_mode              enum     "color"|"blackAndWhite"
    // filterSpeckle  -> filter_speckle          int
    // rdpEpsilon     -> rdp_epsilon             float
    // threshold      -> corner_threshold        float    JS value = "sharpness" angle;
    //                                                    C++ stores 180-threshold as default (120)
    // fitTolerance   -> fit_tolerance           float
    // blurRadius     -> blur_radius             float
    // pathPrecision  -> path_precision          int
    // -------------------------------------------------------------------------
    vtracer::Options opts;

    // ── precision -> color_precision (int) ──────────────────────────────────
    if (params.hasProperty(rt, "precision") && params.getProperty(rt, "precision").isNumber()) {
        opts.color_precision = (int)params.getProperty(rt, "precision").asNumber();
    }

    // ── gradientStep -> gradient_step (float) ───────────────────────────────
    if (params.hasProperty(rt, "gradientStep") && params.getProperty(rt, "gradientStep").isNumber()) {
        opts.gradient_step = (float)params.getProperty(rt, "gradientStep").asNumber();
    }

    // ── colorMode -> color_mode (enum ColorMode) ────────────────────────────
    // JS sends "color" or "blackAndWhite"; map to the C++ enum.
    if (params.hasProperty(rt, "colorMode") && params.getProperty(rt, "colorMode").isString()) {
        std::string colorModeStr = params.getProperty(rt, "colorMode").asString(rt).utf8(rt);
        if (colorModeStr == "blackAndWhite") {
            opts.color_mode = vtracer::ColorMode::BlackAndWhite;
        } else {
            opts.color_mode = vtracer::ColorMode::Color;
        }
    }

    // ── filterSpeckle -> filter_speckle (int) ───────────────────────────────
    if (params.hasProperty(rt, "filterSpeckle") && params.getProperty(rt, "filterSpeckle").isNumber()) {
        opts.filter_speckle = (int)params.getProperty(rt, "filterSpeckle").asNumber();
    }

    // ── rdpEpsilon -> rdp_epsilon (float) ───────────────────────────────────
    if (params.hasProperty(rt, "rdpEpsilon") && params.getProperty(rt, "rdpEpsilon").isNumber()) {
        opts.rdp_epsilon = (float)params.getProperty(rt, "rdpEpsilon").asNumber();
    }

    // ── threshold -> corner_threshold (float) ───────────────────────────────
    // The JS-facing `threshold` is the intuitive "sharpness" angle in degrees.
    // The C++ field `corner_threshold` stores the complementary value
    // (180 - threshold), e.g. JS default 60 => C++ default 120.
    if (params.hasProperty(rt, "threshold") && params.getProperty(rt, "threshold").isNumber()) {
        double threshold = params.getProperty(rt, "threshold").asNumber();
        opts.corner_threshold = (float)(180.0 - threshold);
    }

    // ── fitTolerance -> fit_tolerance (float) ───────────────────────────────
    if (params.hasProperty(rt, "fitTolerance") && params.getProperty(rt, "fitTolerance").isNumber()) {
        opts.fit_tolerance = (float)params.getProperty(rt, "fitTolerance").asNumber();
    }

    // ── blurRadius -> blur_radius (float) ───────────────────────────────────
    if (params.hasProperty(rt, "blurRadius") && params.getProperty(rt, "blurRadius").isNumber()) {
        opts.blur_radius = (float)params.getProperty(rt, "blurRadius").asNumber();
    }

    // ── pathPrecision -> path_precision (int) ───────────────────────────────
    if (params.hasProperty(rt, "pathPrecision") && params.getProperty(rt, "pathPrecision").isNumber()) {
        opts.path_precision = (int)params.getProperty(rt, "pathPrecision").asNumber();
    }

    if (params.hasProperty(rt, "bilateral_sigma_r") && params.getProperty(rt, "bilateral_sigma_r").isNumber()) {
        opts.bilateral_sigma_r = (int)params.getProperty(rt, "bilateral_sigma_r").asNumber();
    }

    if (params.hasProperty(rt, "gradient_detect_thresh") && params.getProperty(rt, "gradient_detect_thresh").isNumber()) {
        opts.gradient_detect_thresh = (int)params.getProperty(rt, "gradient_detect_thresh").asNumber();
    }

    // -------------------------------------------------------------------------
    // Execute vectorization engine
    // -------------------------------------------------------------------------
    std::string svg = vtracer::vectorize(pixels, width, height, opts);

    return jsi::String::createFromUtf8(rt, svg);
}

} // namespace expo::imagetosvg


// -----------------------------------------------------------------------------
// JNI Entry Point
// Called from Kotlin/Java to install the native function into the JS runtime.
// -----------------------------------------------------------------------------
extern "C"
JNIEXPORT void JNICALL
Java_expo_modules_imagetosvg_ExpoImageToSvgModule_installJSIBindings(
    JNIEnv* env,
    jobject /* thiz */,
    jlong jsiRuntimePtr
) {
    // jlong is 64-bit on all Android ABIs that support JSI (arm64-v8a, x86_64),
    // so the reinterpret_cast is safe. Added a null-guard to avoid a hard crash.
    if (jsiRuntimePtr == 0) {
        return; // Nothing to install into; bail out silently.
    }

    auto& runtime = *reinterpret_cast<facebook::jsi::Runtime*>(jsiRuntimePtr);

    auto vectorizeFunc = facebook::jsi::Function::createFromHostFunction(
        runtime,
        facebook::jsi::PropNameID::forAscii(runtime, "nativeVectorize"),
        1,                                  // expected arg count (advisory)
        expo::imagetosvg::vectorizeJSI
    );

    runtime.global().setProperty(runtime, "nativeVectorize", std::move(vectorizeFunc));
}