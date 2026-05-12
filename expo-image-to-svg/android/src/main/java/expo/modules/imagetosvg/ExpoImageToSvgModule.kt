package expo.modules.imagetosvg

import android.util.Log
import com.facebook.react.bridge.ReactApplicationContext
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import kotlin.math.PI

class ExpoImageToSvgModule : Module() {

    companion object {
        private const val TAG = "ExpoImageToSvgModule"

        init {
            try {
                System.loadLibrary("expo-image-to-svg")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load native library: ${e.message}")
            }
        }
    }

    override fun definition() = ModuleDefinition {
        Name("ExpoImageToSvg")

        OnCreate {
            // FIX: cast to ReactApplicationContext, not ReactContext
            val reactContext =
                appContext.reactContext as? ReactApplicationContext ?: return@OnCreate
 
            val holder = reactContext.javaScriptContextHolder ?: return@OnCreate
            
            val jsiRuntimePointer: Long = holder.get()
            if (jsiRuntimePointer != 0L) {
              installJSIBindings(jsiRuntimePointer)
            }
            
        }

        Constant("PI") { PI }   // FIX: kotlin.math.PI (idiomatic)

        Events("onChange")

        Function("hello") {
            "Hello world! 👋"
        }

        AsyncFunction("setValueAsync") { value: String ->
            sendEvent("onChange", mapOf("value" to value))
        }

        View(ExpoImageToSvgView::class) {
            // FIX: use String, not java.net.URL — Expo props don't support URL type
            Prop("url") { view: ExpoImageToSvgView, url: String ->
                view.webView.loadUrl(url)
            }
            Events("onLoad")
        }
    }

    private external fun installJSIBindings(jsiRuntimePointer: Long)
}