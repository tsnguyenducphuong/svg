#include <jni.h>
#include <jsi/jsi.h>
#include "VTracerEngine.hpp"

using namespace facebook;

namespace expo::imagetosvg {

jsi::Value vectorizeJSI(jsi::Runtime& rt, const jsi::Value& thisValue, const jsi::Value* args, size_t count) {
    if (count < 1 || !args[0].isObject()) {
        throw jsi::JSError(rt, "[ExpoImageToSvg] Expected an options object as the first argument.");
    }

    auto params = args[0].asObject(rt);

    // 1. Extract Buffer (Handling Uint8Array -> ArrayBuffer)
    // In JS, Uint8Array has a .buffer property which is the actual ArrayBuffer
    auto typedArray = params.getPropertyAsObject(rt, "buffer");
    auto arrayBuffer = typedArray.getPropertyAsObject(rt, "buffer").getArrayBuffer(rt);
    uint8_t* pixels = arrayBuffer.data(rt);

    // 2. Extract Dimensions
    int width = (int)params.getProperty(rt, "width").asNumber();
    int height = (int)params.getProperty(rt, "height").asNumber();

    if (width <= 0 || height <= 0) {
        throw jsi::JSError(rt, "[ExpoImageToSvg] Invalid image dimensions.");
    }

    // 3. Map Parameters to vtracer::Options
    vtracer::Options opts;
    
    // JS: precision -> C++: color_precision
    if (params.hasProperty(rt, "precision")) {
        opts.color_precision = (int)params.getProperty(rt, "precision").asNumber();
    }
    
    // JS: threshold -> C++: corner_threshold (Degrees)
    if (params.hasProperty(rt, "threshold")) {
        opts.corner_threshold = (float)params.getProperty(rt, "threshold").asNumber();
    }
    
    // JS: filterSpeckle -> C++: filter_speckle (Pixel count)
    if (params.hasProperty(rt, "filterSpeckle")) {
        opts.filter_speckle = (int)params.getProperty(rt, "filterSpeckle").asNumber();
    }

    // 4. Execute Engine
    std::string svg = vtracer::vectorize(pixels, width, height, opts);
    
    return jsi::String::createFromUtf8(rt, svg);
}

} // namespace expo::imagetosvg

// JNI Entry Point
extern "C"
JNIEXPORT void JNICALL
Java_expo_modules_imagetosvg_ExpoImageToSvgModule_installJSIBindings(JNIEnv* env, jobject thiz, jlong jsiRuntimePtr) {
    auto& runtime = *(facebook::jsi::Runtime*)jsiRuntimePtr;

    auto vectorizeFunc = facebook::jsi::Function::createFromHostFunction(
        runtime,
        facebook::jsi::PropNameID::forAscii(runtime, "nativeVectorize"),
        1,
        expo::imagetosvg::vectorizeJSI
    );

    runtime.global().setProperty(runtime, "nativeVectorize", std::move(vectorizeFunc));
}
