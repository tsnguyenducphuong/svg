#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface ExpoImageToSvgJSIInstaller : NSObject

/**
 * Registers the 'nativeVectorize' function into the global JSI runtime.
 * @param jsiRuntimePtr An opaque pointer to the facebook::jsi::Runtime.
 */
+ (void)install:(void *)jsiRuntimePtr;

@end

NS_ASSUME_NONNULL_END