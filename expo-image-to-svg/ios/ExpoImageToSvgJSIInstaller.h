#pragma once
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface ExpoImageToSvgJSIInstaller : NSObject
/// Equivalent to Android's installJSIBindings(jsiRuntimePointer: Long)
+ (void)install:(void *)runtimePointer;
@end

NS_ASSUME_NONNULL_END