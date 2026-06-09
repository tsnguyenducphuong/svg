#import <Foundation/Foundation.h>
#import <React/RCTBridge.h>

NS_ASSUME_NONNULL_BEGIN

@interface ExpoImageToSvgJSIInstaller : NSObject

/**
 * Registers the 'nativeVectorize' and 'nativeVectorizeMultiPass' functions
 * into the global JSI runtime.
 * @param bridge The RCTBridge instance from appContext.reactBridge.
 *               Internally casts to RCTCxxBridge and reads javaScriptContextHolder
 *               to reach the underlying jsi::Runtime — compatible with SDK 54+.
 */
+ (void)installWithBridge:(RCTBridge *)bridge;

@end

NS_ASSUME_NONNULL_END