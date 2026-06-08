#import <Foundation/Foundation.h>
#import <ExpoModulesCore/EXJavaScriptRuntime.h>

NS_ASSUME_NONNULL_BEGIN

@interface ExpoImageToSvgJSIInstaller : NSObject

/**
 * Registers the 'nativeVectorize' and 'nativeVectorizeMultiPass' functions
 * into the global JSI runtime.
 * @param runtimeObj The EXJavaScriptRuntime instance from appContext.runtime.
 */
+ (void)install:(id)runtimeObj;

@end

NS_ASSUME_NONNULL_END