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
    // FIX 1: Correct ArrayBuffer extraction from a Uint8Array.
    //
    // Previously the code called:
    //   typedArray.getPropertyAsObject(rt, "buffer").getArrayBuffer(rt)
    // which went one level too deep (buffer.buffer).
    //
    // A JS Uint8Array is NOT an ArrayBuffer itself. It is a typed-array object
    // whose underlying storage lives in its ".buffer" property (an ArrayBuffer).
    //
    // Correct approach:
    //   1. Get the Uint8Array object from params ("buffer" key on the JS side).
    //   2. Read its ".buffer" property to obtain the raw ArrayBuffer.
    //   3. Verify it really is an ArrayBuffer before calling getArrayBuffer().
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

    // FIX 2: Use asArrayBuffer() (not getArrayBuffer()) — the JSI method on
    // jsi::Object that returns a jsi::ArrayBuffer when you already know it is one.
    auto arrayBuffer = rawBufferVal.asObject(rt).getArrayBuffer(rt);
    uint8_t* pixels = arrayBuffer.data(rt);
    size_t byteLength = arrayBuffer.size(rt);

    // -------------------------------------------------------------------------
    // Extract & validate dimensions
    // -------------------------------------------------------------------------

    // FIX 3: Guard against missing or non-number properties before calling
    // asNumber(), which throws a hard JSI error on type mismatch.
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

    // Optional: sanity-check that the buffer is large enough for the declared
    // dimensions (RGBA = 4 bytes per pixel).
    if (byteLength < (size_t)(width * height * 4)) {
        throw jsi::JSError(rt, "[ExpoImageToSvg] Buffer is too small for the given dimensions.");
    }

    // -------------------------------------------------------------------------
    // Map JS options -> vtracer::Options
    // -------------------------------------------------------------------------
    vtracer::Options opts;

    // JS: precision -> C++: color_precision  (int)
    if (params.hasProperty(rt, "precision") && params.getProperty(rt, "precision").isNumber()) {
        opts.color_precision = (int)params.getProperty(rt, "precision").asNumber();
    }

    // JS: threshold -> C++: corner_threshold
    // FIX 4: vtracer exposes corner_threshold as a double; keep full precision
    // and avoid the implicit narrowing float cast that could trigger a compiler
    // warning (or silent precision loss on some toolchains).
    if (params.hasProperty(rt, "threshold") && params.getProperty(rt, "threshold").isNumber()) {
        opts.corner_threshold = params.getProperty(rt, "threshold").asNumber();
    }

    // JS: filterSpeckle -> C++: filter_speckle  (int)
    if (params.hasProperty(rt, "filterSpeckle") && params.getProperty(rt, "filterSpeckle").isNumber()) {
        opts.filter_speckle = (int)params.getProperty(rt, "filterSpeckle").asNumber();
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
    // FIX 5: Cast jsiRuntimePtr safely. jlong is 64-bit on all Android ABIs
    // that support JSI (arm64-v8a, x86_64), so the reinterpret_cast is safe.
    // Added a null-guard to avoid a hard crash if a bad pointer is passed.
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