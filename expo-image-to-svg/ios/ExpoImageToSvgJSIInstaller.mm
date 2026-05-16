#import "ExpoImageToSvgJSIInstaller.h"
#import <jsi/jsi.h>
#import "VTracerEngine.hpp"




using namespace facebook;




namespace expo::imagetosvg {




// =============================================================================
//  Helpers
// =============================================================================




// Unwrap a JS Uint8Array value into a raw uint8_t pointer + byte length.
// Throws a jsi::JSError on any type mismatch.
static uint8_t* unwrapUint8Array(
   jsi::Runtime& rt,
   const jsi::Object& params,
   const char* key,
   size_t& outByteLength)
{
   if (!params.hasProperty(rt, key)) {
       throw jsi::JSError(rt,
           std::string("[ExpoImageToSvg] Missing required property: ") + key + ".");
   }


   auto typedArrayObj = params.getPropertyAsObject(rt, key);


   if (!typedArrayObj.hasProperty(rt, "buffer")) {
       throw jsi::JSError(rt,
           std::string("[ExpoImageToSvg] Property \"") + key +
           "\" is not a valid Uint8Array (missing .buffer).");
   }


   auto rawBufferVal = typedArrayObj.getProperty(rt, "buffer");
   if (!rawBufferVal.isObject() || !rawBufferVal.asObject(rt).isArrayBuffer(rt)) {
       throw jsi::JSError(rt,
           std::string("[ExpoImageToSvg] \"") + key + "\".buffer is not an ArrayBuffer.");
   }


   auto arrayBuffer = rawBufferVal.asObject(rt).getArrayBuffer(rt);
   outByteLength = arrayBuffer.size(rt);
   return arrayBuffer.data(rt);
}




// Read an optional float property. Returns defaultValue if absent or non-number.
static float getOptFloat(
   jsi::Runtime& rt,
   const jsi::Object& obj,
   const char* key,
   float defaultValue)
{
   if (obj.hasProperty(rt, key) && obj.getProperty(rt, key).isNumber())
       return (float)obj.getProperty(rt, key).asNumber();
   return defaultValue;
}




// Read an optional int property. Returns defaultValue if absent or non-number.
static int getOptInt(
   jsi::Runtime& rt,
   const jsi::Object& obj,
   const char* key,
   int defaultValue)
{
   if (obj.hasProperty(rt, key) && obj.getProperty(rt, key).isNumber())
       return (int)obj.getProperty(rt, key).asNumber();
   return defaultValue;
}




// Read an optional bool property. Returns defaultValue if absent or non-bool.
static bool getOptBool(
   jsi::Runtime& rt,
   const jsi::Object& obj,
   const char* key,
   bool defaultValue)
{
   if (obj.hasProperty(rt, key) && obj.getProperty(rt, key).isBool())
       return obj.getProperty(rt, key).asBool();
   return defaultValue;
}




// ---------------------------------------------------------------------------
//  Map a JS PassOptions object → vtracer::Options.
//  Used for pass1, pass2, pass3 (and the reserved pass4/pass5) inside
//  vectorizeMultiPassJSI.
//
//  JS key               C++ field                    Notes
//  ─────────────────    ────────────────────────     ──────────────────────
//  precision         →  color_precision              int
//  segmentThreshold  →  segment_threshold            float
//  threshold         →  corner_threshold             float (180 - threshold)
//  filterSpeckle     →  filter_speckle               int
//  rdpEpsilon        →  rdp_epsilon                  float
//  fitTolerance      →  fit_tolerance                float
//  pathPrecision     →  path_precision               int
//  gradientDetectThresh → gradient_detect_thresh     float
//  bilateralSigmaR   →  bilateral_sigma_r            float
//  blurRadius        →  blur_radius                  float
//  colorMode         →  color_mode                   enum
// ---------------------------------------------------------------------------
static vtracer::Options parsePassOptions(
   jsi::Runtime& rt,
   const jsi::Object& passObj,
   const vtracer::Options& defaults)
{
   vtracer::Options o = defaults;


   if (passObj.hasProperty(rt, "precision") &&
       passObj.getProperty(rt, "precision").isNumber())
       o.color_precision = (int)passObj.getProperty(rt, "precision").asNumber();


   // if (passObj.hasProperty(rt, "segmentThreshold") &&
   //     passObj.getProperty(rt, "segmentThreshold").isNumber())
   //     o.segment_threshold = (float)passObj.getProperty(rt, "segmentThreshold").asNumber();


   if (passObj.hasProperty(rt, "threshold") &&
       passObj.getProperty(rt, "threshold").isNumber())
       o.corner_threshold = (float)(180.0 - passObj.getProperty(rt, "threshold").asNumber());


   if (passObj.hasProperty(rt, "filterSpeckle") &&
       passObj.getProperty(rt, "filterSpeckle").isNumber())
       o.filter_speckle = (int)passObj.getProperty(rt, "filterSpeckle").asNumber();


   if (passObj.hasProperty(rt, "rdpEpsilon") &&
       passObj.getProperty(rt, "rdpEpsilon").isNumber())
       o.rdp_epsilon = (float)passObj.getProperty(rt, "rdpEpsilon").asNumber();


   if (passObj.hasProperty(rt, "fitTolerance") &&
       passObj.getProperty(rt, "fitTolerance").isNumber())
       o.fit_tolerance = (float)passObj.getProperty(rt, "fitTolerance").asNumber();


   if (passObj.hasProperty(rt, "pathPrecision") &&
       passObj.getProperty(rt, "pathPrecision").isNumber())
       o.path_precision = (int)passObj.getProperty(rt, "pathPrecision").asNumber();


   if (passObj.hasProperty(rt, "gradientDetectThresh") &&
       passObj.getProperty(rt, "gradientDetectThresh").isNumber())
       o.gradient_detect_thresh = (float)passObj.getProperty(rt, "gradientDetectThresh").asNumber();


   if (passObj.hasProperty(rt, "bilateralSigmaR") &&
       passObj.getProperty(rt, "bilateralSigmaR").isNumber())
       o.bilateral_sigma_r = (float)passObj.getProperty(rt, "bilateralSigmaR").asNumber();


   if (passObj.hasProperty(rt, "blurRadius") &&
       passObj.getProperty(rt, "blurRadius").isNumber())
       o.blur_radius = (float)passObj.getProperty(rt, "blurRadius").asNumber();


   if (passObj.hasProperty(rt, "colorMode") &&
       passObj.getProperty(rt, "colorMode").isString()) {
       std::string cm = passObj.getProperty(rt, "colorMode").asString(rt).utf8(rt);
       o.color_mode = (cm == "blackAndWhite")
           ? vtracer::ColorMode::BlackAndWhite
           : vtracer::ColorMode::Color;
   }


   return o;
}




// =============================================================================
//  JSI host function: nativeVectorize  (single-pass)
// =============================================================================
jsi::Value vectorizeJSI(
   jsi::Runtime& rt,
   const jsi::Value& /*thisValue*/,
   const jsi::Value* args,
   size_t count)
{
   if (count < 1 || !args[0].isObject())
       throw jsi::JSError(rt, "[ExpoImageToSvg] Expected an options object.");


   auto params = args[0].asObject(rt);


   // ── Buffer ───────────────────────────────────────────────────────────────
   size_t byteLength = 0;
   uint8_t* pixels = unwrapUint8Array(rt, params, "buffer", byteLength);


   // ── Dimensions ───────────────────────────────────────────────────────────
   if (!params.hasProperty(rt, "width") || !params.getProperty(rt, "width").isNumber())
       throw jsi::JSError(rt, "[ExpoImageToSvg] Missing or invalid property: width.");
   if (!params.hasProperty(rt, "height") || !params.getProperty(rt, "height").isNumber())
       throw jsi::JSError(rt, "[ExpoImageToSvg] Missing or invalid property: height.");


   int width  = (int)params.getProperty(rt, "width").asNumber();
   int height = (int)params.getProperty(rt, "height").asNumber();


   if (width <= 0 || height <= 0)
       throw jsi::JSError(rt, "[ExpoImageToSvg] Invalid image dimensions.");


   if (byteLength < (size_t)(width * height * 4))
       throw jsi::JSError(rt, "[ExpoImageToSvg] Buffer is too small for the given dimensions.");


   // ── Options ──────────────────────────────────────────────────────────────
   //
   // JS key               C++ field                Notes
   // ─────────────────    ──────────────────────── ─────────────────────────
   // precision         →  color_precision           int
   // gradientDetectThresh → gradient_detect_thresh  float
   // colorMode         →  color_mode                enum     "color"|"blackAndWhite"
   // filterSpeckle     →  filter_speckle            int
   // rdpEpsilon        →  rdp_epsilon               float
   // threshold         →  corner_threshold          float    180 - threshold
   // fitTolerance      →  fit_tolerance             float
   // blurRadius        →  blur_radius               float
   // pathPrecision     →  path_precision            int
   // bilateralSigmaR   →  bilateral_sigma_r         float
   // segmentThreshold  →  segment_threshold         float
   // -------------------------------------------------------------------------
   vtracer::Options opts;


   if (params.hasProperty(rt, "precision") && params.getProperty(rt, "precision").isNumber())
       opts.color_precision = (int)params.getProperty(rt, "precision").asNumber();


   if (params.hasProperty(rt, "gradientDetectThresh") && params.getProperty(rt, "gradientDetectThresh").isNumber())
       opts.gradient_detect_thresh = (float)params.getProperty(rt, "gradientDetectThresh").asNumber();


   if (params.hasProperty(rt, "colorMode") && params.getProperty(rt, "colorMode").isString()) {
       std::string cm = params.getProperty(rt, "colorMode").asString(rt).utf8(rt);
       opts.color_mode = (cm == "blackAndWhite")
           ? vtracer::ColorMode::BlackAndWhite
           : vtracer::ColorMode::Color;
   }


   if (params.hasProperty(rt, "filterSpeckle") && params.getProperty(rt, "filterSpeckle").isNumber())
       opts.filter_speckle = (int)params.getProperty(rt, "filterSpeckle").asNumber();


   if (params.hasProperty(rt, "rdpEpsilon") && params.getProperty(rt, "rdpEpsilon").isNumber())
       opts.rdp_epsilon = (float)params.getProperty(rt, "rdpEpsilon").asNumber();


   // JS `threshold` is the intuitive sharpness angle; C++ stores (180 - threshold).
   if (params.hasProperty(rt, "threshold") && params.getProperty(rt, "threshold").isNumber())
       opts.corner_threshold = (float)(180.0 - params.getProperty(rt, "threshold").asNumber());


   if (params.hasProperty(rt, "fitTolerance") && params.getProperty(rt, "fitTolerance").isNumber())
       opts.fit_tolerance = (float)params.getProperty(rt, "fitTolerance").asNumber();


   if (params.hasProperty(rt, "blurRadius") && params.getProperty(rt, "blurRadius").isNumber())
       opts.blur_radius = (float)params.getProperty(rt, "blurRadius").asNumber();


   if (params.hasProperty(rt, "pathPrecision") && params.getProperty(rt, "pathPrecision").isNumber())
       opts.path_precision = (int)params.getProperty(rt, "pathPrecision").asNumber();


   if (params.hasProperty(rt, "bilateralSigmaR") && params.getProperty(rt, "bilateralSigmaR").isNumber())
       opts.bilateral_sigma_r = (float)params.getProperty(rt, "bilateralSigmaR").asNumber();


   // if (params.hasProperty(rt, "segmentThreshold") && params.getProperty(rt, "segmentThreshold").isNumber())
   //     opts.segment_threshold = (float)params.getProperty(rt, "segmentThreshold").asNumber();


   // ── Execute ──────────────────────────────────────────────────────────────
   std::string svg = vtracer::vectorize(pixels, width, height, opts);
   return jsi::String::createFromUtf8(rt, svg);
}




// =============================================================================
//  JSI host function: nativeVectorizeMultiPass  (ENH-12 6-pass pipeline)
//
//  JS options object layout (all buffer keys hold Uint8Arrays):
//
//  Required:
//    originalBuffer       Uint8Array  — source RGBA pixels
//    blurBuffer           Uint8Array  — Gaussian/bilateral blurred RGBA
//    highPassBuffer       Uint8Array  — high-pass residual RGBA
//                                       each ch = clamp(orig - blur + 128, 0, 255)
//    maskBuffer           Uint8Array  — subject mask (R ≥ 128 = fg) RGBA
//    edgeMapBuffer        Uint8Array  — Canny/Sobel edge map (R = strength) RGBA
//    width                number
//    height               number
//
//  Top-level compositing (all optional):
//    baseDilateRadius       number   → mpOpts.baseDilateRadius     (default 2.0)
//    edgeStrokeWidth        number   → mpOpts.edgeStrokeWidth       (default 0.5)
//    edgeMinLuminance       number   → mpOpts.edgeMinLuminance      (default 80)
//    highPassGroupOpacity   number   → mpOpts.highPassGroupOpacity  (legacy compat)
//
//  ENH-12a  Local Color Quantization (Pass 2):
//    lcqGridW               number   → mpOpts.lcqGridW              (default 16)
//    lcqGridH               number   → mpOpts.lcqGridH              (default 16)
//    lcqColorsPerTile       number   → mpOpts.lcqColorsPerTile      (default 24)
//
//  ENH-12b  Adaptive Threshold (Pass 3):
//    microDetailDeltaEThresh number  → mpOpts.microDetailDeltaEThresh (default 6.0)
//
//  ENH-12c  Highlight / Shadow extraction (Passes 4 & 5):
//    highlightLStar         number   → mpOpts.highlightLStar        (default 85.0)
//    shadowLStar            number   → mpOpts.shadowLStar           (default 28.0)
//
//  Per-pass sub-objects (all optional — engine defaults are pre-configured):
//    pass1   PassOptions  → base/blur layer tuning
//    pass2   PassOptions  → mid-tones / LCQ layer tuning
//    pass3   PassOptions  → micro-detail layer tuning
//    pass4   PassOptions  → highlights layer tuning (reserved, currently ignored)
//    pass5   PassOptions  → low-lights layer tuning (reserved, currently ignored)
// =============================================================================
jsi::Value vectorizeMultiPassJSI(
   jsi::Runtime& rt,
   const jsi::Value& /*thisValue*/,
   const jsi::Value* args,
   size_t count)
{
   if (count < 1 || !args[0].isObject())
       throw jsi::JSError(rt, "[ExpoImageToSvg] vectorizeMultiPass: Expected an options object.");


   auto params = args[0].asObject(rt);


   // ── Dimensions ───────────────────────────────────────────────────────────
   if (!params.hasProperty(rt, "width") || !params.getProperty(rt, "width").isNumber())
       throw jsi::JSError(rt, "[ExpoImageToSvg] vectorizeMultiPass: Missing or invalid property: width.");
   if (!params.hasProperty(rt, "height") || !params.getProperty(rt, "height").isNumber())
       throw jsi::JSError(rt, "[ExpoImageToSvg] vectorizeMultiPass: Missing or invalid property: height.");


   int width  = (int)params.getProperty(rt, "width").asNumber();
   int height = (int)params.getProperty(rt, "height").asNumber();


   if (width <= 0 || height <= 0)
       throw jsi::JSError(rt, "[ExpoImageToSvg] vectorizeMultiPass: Invalid image dimensions.");


   const size_t expectedBytes = (size_t)(width * height * 4);


   // ── Unwrap all five buffers ───────────────────────────────────────────────
   size_t bl0, bl1, bl2, bl3, bl4;
   uint8_t* originalPixels  = unwrapUint8Array(rt, params, "originalBuffer",  bl0);
   uint8_t* blurPixels      = unwrapUint8Array(rt, params, "blurBuffer",      bl1);
   uint8_t* highPassPixels  = unwrapUint8Array(rt, params, "highPassBuffer",  bl2);
   uint8_t* maskPixels      = unwrapUint8Array(rt, params, "maskBuffer",      bl3);
   uint8_t* edgeMapPixels   = unwrapUint8Array(rt, params, "edgeMapBuffer",   bl4);


   // Sanity-check all buffer sizes.
   const char* bufKeys[] = {
       "originalBuffer", "blurBuffer", "highPassBuffer",
       "maskBuffer",     "edgeMapBuffer"
   };
   size_t bufLens[] = { bl0, bl1, bl2, bl3, bl4 };
   for (int i = 0; i < 5; ++i) {
       if (bufLens[i] < expectedBytes) {
           throw jsi::JSError(rt,
               std::string("[ExpoImageToSvg] vectorizeMultiPass: \"") + bufKeys[i] +
               "\" is too small for the given dimensions.");
       }
   }


   // ── Build MultiPassOptions ────────────────────────────────────────────────
   // Start with struct defaults (all matching the C++ header defaults).
   vtracer::MultiPassOptions mpOpts;


   // ── Top-level compositing controls ───────────────────────────────────────


   // ENH-12d: Variable dilation — base layer seals gaps with 2 px dilation.
   mpOpts.baseDilateRadius =
       getOptFloat(rt, params, "baseDilateRadius", mpOpts.baseDilateRadius);


   // Pass 6 (Edge/Ink) stroke parameters.
   mpOpts.edgeStrokeWidth =
       getOptFloat(rt, params, "edgeStrokeWidth",  mpOpts.edgeStrokeWidth);
   mpOpts.edgeMinLuminance =
       getOptInt  (rt, params, "edgeMinLuminance", mpOpts.edgeMinLuminance);


   // Legacy field kept for source compatibility with existing callers.
   mpOpts.highPassGroupOpacity =
       getOptFloat(rt, params, "highPassGroupOpacity", mpOpts.highPassGroupOpacity);


   // ── ENH-12a: Local Color Quantization grid (Pass 2) ───────────────────────
   // Controls the 16×16 tile partition and per-tile palette depth.
   mpOpts.lcqGridW =
       getOptInt(rt, params, "lcqGridW",         mpOpts.lcqGridW);
   mpOpts.lcqGridH =
       getOptInt(rt, params, "lcqGridH",         mpOpts.lcqGridH);
   mpOpts.lcqColorsPerTile =
       getOptInt(rt, params, "lcqColorsPerTile", mpOpts.lcqColorsPerTile);


   // ── ENH-12b: Adaptive Threshold gate (Pass 3) ─────────────────────────────
   // Pixels whose high-pass colour is perceptually too close (ΔE < threshold)
   // to the underlying Pass-2 colour are suppressed, keeping file size earned.
   mpOpts.microDetailDeltaEThresh =
       getOptFloat(rt, params, "microDetailDeltaEThresh", mpOpts.microDetailDeltaEThresh);


   // ── ENH-12c: Highlight / Shadow L* thresholds (Passes 4 & 5) ─────────────
   // Pass 4 extracts pixels above highlightLStar (top ~10% brightness).
   // Pass 5 extracts pixels below shadowLStar (darkest ~15%).
   mpOpts.highlightLStar =
       getOptFloat(rt, params, "highlightLStar", mpOpts.highlightLStar);
   mpOpts.shadowLStar =
       getOptFloat(rt, params, "shadowLStar",    mpOpts.shadowLStar);


   // ── Per-pass options ──────────────────────────────────────────────────────
   // Each pass sub-object is merged over the C++ struct defaults, so only the
   // fields explicitly provided by the caller are overridden.


   // Pass 1 — Base / blur layer (large smooth painterly fills).
   if (params.hasProperty(rt, "pass1") && params.getProperty(rt, "pass1").isObject())
       mpOpts.pass1 = parsePassOptions(rt, params.getPropertyAsObject(rt, "pass1"), mpOpts.pass1);


   // Pass 2 — Mid-Tones / Local Color Quantization layer.
   if (params.hasProperty(rt, "pass2") && params.getProperty(rt, "pass2").isObject())
       mpOpts.pass2 = parsePassOptions(rt, params.getPropertyAsObject(rt, "pass2"), mpOpts.pass2);


   // Pass 3 — Micro-Detail / Adaptive Threshold high-pass layer.
   if (params.hasProperty(rt, "pass3") && params.getProperty(rt, "pass3").isObject())
       mpOpts.pass3 = parsePassOptions(rt, params.getPropertyAsObject(rt, "pass3"), mpOpts.pass3);


   // Pass 4 — Highlights (reserved; engine uses hardcoded optimal values).
   if (params.hasProperty(rt, "pass4") && params.getProperty(rt, "pass4").isObject())
       mpOpts.pass4 = parsePassOptions(rt, params.getPropertyAsObject(rt, "pass4"), mpOpts.pass4);


   // Pass 5 — Low-Lights / Shadows (reserved; engine uses hardcoded optimal values).
   if (params.hasProperty(rt, "pass5") && params.getProperty(rt, "pass5").isObject())
       mpOpts.pass5 = parsePassOptions(rt, params.getPropertyAsObject(rt, "pass5"), mpOpts.pass5);


   // ── Execute ──────────────────────────────────────────────────────────────
   std::string svg = vtracer::vectorizeMultiPass(
       originalPixels,
       blurPixels,
       highPassPixels,
       maskPixels,
       edgeMapPixels,
       width, height,
       mpOpts);


   return jsi::String::createFromUtf8(rt, svg);
}




} // namespace expo::imagetosvg




// =============================================================================
//  Objective-C entry point
//  Called from Swift/ObjC to install both native functions into the JS runtime.
// =============================================================================
@implementation ExpoImageToSvgJSIInstaller


+ (void)install:(void *)jsiRuntimePtr {
   if (!jsiRuntimePtr) return;


   jsi::Runtime &rt = *(jsi::Runtime *)jsiRuntimePtr;


   // ── nativeVectorize (single-pass) ────────────────────────────────────────
   auto vectorizeFunc = jsi::Function::createFromHostFunction(
       rt,
       jsi::PropNameID::forAscii(rt, "nativeVectorize"),
       1,
       expo::imagetosvg::vectorizeJSI);
   rt.global().setProperty(rt, "nativeVectorize", std::move(vectorizeFunc));


   // ── nativeVectorizeMultiPass (ENH-12 6-pass Stochastic Painterly) ────────
   auto multiPassFunc = jsi::Function::createFromHostFunction(
       rt,
       jsi::PropNameID::forAscii(rt, "nativeVectorizeMultiPass"),
       1,
       expo::imagetosvg::vectorizeMultiPassJSI);
   rt.global().setProperty(rt, "nativeVectorizeMultiPass", std::move(multiPassFunc));
}


@end