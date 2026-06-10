require 'json'

package = JSON.parse(File.read(File.join(__dir__, '..', 'package.json')))

Pod::Spec.new do |s|
  s.name           = 'ExpoImageToSvg'
  s.version        = package['version']
  s.summary        = package['description']
  s.description    = package['description']
  s.license        = package['license']
  s.author         = package['author']
  s.homepage       = package['homepage']
  s.platforms      = {
    :ios => '15.1',
    :tvos => '15.1'
  }

  # Swift version must be set via s.swift_version, NOT via pod_target_xcconfig.
  # Setting it in xcconfig is silently ignored by CocoaPods.
  s.swift_version  = '5.9'

  s.source         = { git: 'https://github.com/tsnguyenducphuong/svg' }
  s.static_framework = true

  s.dependency 'ExpoModulesCore'
  s.dependency 'React-jsi'

  s.source_files = "**/*.{h,m,mm,swift}", "../cpp/**/*.{hpp,cpp}"

  # Single pod_target_xcconfig block.
  # NOTE: pod_target_xcconfig must only be assigned ONCE in Ruby — a second
  # assignment silently overwrites the first (which is why DEFINES_MODULE was
  # being lost in the original file). All settings are merged here.
  #
  # SWIFT_COMPILATION_MODE is intentionally removed:
  #   - 'wholemodule' changes how Swift resolves cross-module types at compile
  #     time and is known to break visibility of ExpoModulesCore members
  #     (including AppContext.runtime) when the dependency is compiled with a
  #     different mode. CocoaPods / Xcode manages this automatically.
  s.pod_target_xcconfig = {
    'DEFINES_MODULE'              => 'YES',
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'CLANG_CXX_LIBRARY'           => 'libc++',
    'HEADER_SEARCH_PATHS'         => [
      '"$(PODS_TARGET_SRCROOT)/../cpp"',
      '"$(PODS_ROOT)/Headers/Public/React-Core"',
      '"$(PODS_ROOT)/Headers/Public/React-hermes"',
      '"$(PODS_ROOT)/Headers/Public/React-jsi"'
    ].join(' ')
  }
end
