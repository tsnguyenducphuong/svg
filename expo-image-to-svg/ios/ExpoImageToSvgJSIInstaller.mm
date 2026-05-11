#import "ExpoImageToSvgJSIInstaller.h"
#import <jsi/jsi.h>
#import "VTracerEngine.hpp"

using namespace facebook;

@implementation ExpoImageToSvgJSIInstaller

+ (void)install:(void *)jsiRuntimePtr {
    if (!jsiRuntimePtr) return;
    
    jsi::Runtime &rt = *(jsi::Runtime *)jsiRuntimePtr;

    auto vectorizeFunc = jsi::Function::createFromHostFunction(
        rt,
        jsi::PropNameID::forAscii(rt, "nativeVectorize"),
        1,
        [](jsi::Runtime &rt, const jsi::Value &thisValue, const jsi::Value *args, size_t count) -> jsi::Value {
            if (count < 1 || !args[0].isObject()) {
                throw jsi::JSError(rt, "Expected an options object as the first argument.");
            }

            auto params = args[0].asObject(rt);
            auto bufferObj = params.getPropertyAsObject(rt, "buffer");
            auto arrayBuffer = bufferObj.getPropertyAsObject(rt, "buffer").getArrayBuffer(rt);

            // Direct memory pointer - zero copy
            uint8_t* pixels = arrayBuffer.data(rt);
            int width = (int)params.getProperty(rt, "width").asNumber();
            int height = (int)params.getProperty(rt, "height").asNumber();

            vtracer::Options opts;
            opts.color_precision = (float)params.getProperty(rt, "precision").asNumber();
            opts.corner_threshold = (float)params.getProperty(rt, "threshold").asNumber();
            opts.filter_speckle = (int)params.getProperty(rt, "filterSpeckle").asNumber();

            // Execute the highly-optimized C++ engine
            std::string svg = vtracer::vectorize(pixels, width, height, opts);
            
            return jsi::String::createFromUtf8(rt, svg);
        }
    );

    // Inject function into the global JS context
    rt.global().setProperty(rt, "nativeVectorize", std::move(vectorizeFunc));
}

@end