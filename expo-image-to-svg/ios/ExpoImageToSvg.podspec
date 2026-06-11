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

  # FIX 1: Remove s.static_framework = true
  # Combining static_framework with DEFINES_MODULE = YES creates a contradictory
  # configuration. A static framework doesn't vend a proper module map, so the
  # Swift compiler sees a "hollow" AppContext type and can't resolve its members
  # (including .runtime). CocoaPods handles linkage automatically via the
  # :linkage setting in the app's Podfile; the pod itself should not force it.

  s.dependency 'ExpoModulesCore'
  s.dependency 'React-jsi'

  s.source_files = "**/*.{h,m,mm,swift}", "../cpp/**/*.{hpp,cpp}"

  # Single pod_target_xcconfig block — assigned exactly ONCE.
  #
  # FIX 2: Remove DEFINES_MODULE = YES
  # This is the primary cause of "AppContext has no member 'runtime'".
  # DEFINES_MODULE changes how Swift resolves cross-module types at compile time.
  # When your pod declares itself as a module, the Swift compiler builds a
  # module boundary between it and ExpoModulesCore. AppContext's Swift members
  # (which are @_spi or have access modifiers tied to module identity) become
  # invisible across that boundary — the linker finds them but the type-checker
  # doesn't. Removing this setting lets CocoaPods manage module exposure the
  # standard way, which is what ExpoModulesCore expects its consumers to do.
  #
  # FIX 3: Add ExpoModulesCore to HEADER_SEARCH_PATHS
  # Without this, the Swift compiler can't fully resolve AppContext's interface
  # when compiling the mixed Swift/.mm source files in this pod.
  #
  # NOTE: SWIFT_COMPILATION_MODE is intentionally absent.
  # 'wholemodule' is known to break visibility of ExpoModulesCore members
  # (including AppContext.runtime) when the dependency uses a different mode.
  # Xcode manages compilation mode automatically per build configuration.
  s.pod_target_xcconfig = {
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'CLANG_CXX_LIBRARY'           => 'libc++',
    # Enable C++ Interoperability for the Swift compiler
    'OTHER_SWIFT_FLAGS' => '-cxx-interoperability-mode=default',
    'HEADER_SEARCH_PATHS'         => [
      '"$(PODS_TARGET_SRCROOT)/../cpp"',
      '"$(PODS_ROOT)/Headers/Public/React-Core"',
      '"$(PODS_ROOT)/Headers/Public/React-hermes"',
      '"$(PODS_ROOT)/Headers/Public/React-jsi"',
      # FIX 3: ExpoModulesCore headers must be visible so the Swift compiler
      # can fully resolve AppContext's interface in the mixed-source compilation.
      '"$(PODS_ROOT)/Headers/Public/ExpoModulesCore"'
    ].join(' ')
  }
end
