#import <Foundation/Foundation.h>
#import <ExpoModulesCore/EXJavaScriptRuntime.h>

NS_ASSUME_NONNULL_BEGIN

@interface ExpoImageToSvgJSIInstaller : NSObject

/**
 * Registers the 'nativeVectorize' and 'nativeVectorizeMultiPass' functions
 * into the global JSI runtime.
 * @param runtime The EXJavaScriptRuntime (Swift name: JavaScriptRuntime) obtained
 *                from `try? appContext?.runtime` inside OnCreate. This is the
 *                correct SDK 54 approach — AppContext.runtime is a throwing
 *                computed property returning ExpoRuntime (a JavaScriptRuntime subclass).
 */
+ (void)install:(EXJavaScriptRuntime *)runtime;

@end

NS_ASSUME_NONNULL_END