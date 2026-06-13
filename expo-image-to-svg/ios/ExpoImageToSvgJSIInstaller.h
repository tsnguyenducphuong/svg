#pragma once
#import <Foundation/Foundation.h>
#import <ExpoModulesCore/ExpoModulesCore-Swift.h> // exposes JavaScriptRuntime to ObjC

NS_ASSUME_NONNULL_BEGIN

@interface ExpoImageToSvgJSIInstaller : NSObject
+ (void)install:(JavaScriptRuntime *)runtime; // ← Expo wrapper, not raw pointer
@end

NS_ASSUME_NONNULL_END