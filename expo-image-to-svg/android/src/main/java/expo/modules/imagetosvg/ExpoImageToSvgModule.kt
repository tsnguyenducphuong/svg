package expo.modules.imagetosvg

import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import java.net.URL
import com.facebook.react.bridge.ReactApplicationContext

class ExpoImageToSvgModule : Module() {

  init {
    try {
      System.loadLibrary("expo-image-to-svg")
    } catch (e: UnsatisfiedLinkError) {
      e.printStackTrace()
    }
  }

  override fun definition() = ModuleDefinition {

    Name("ExpoImageToSvg")

    OnCreate {
      val reactContext =
        appContext.reactContext as? ReactApplicationContext

      val catalystInstance = reactContext?.catalystInstance

      if (catalystInstance == null || catalystInstance.isDestroyed) {
        return@OnCreate
      }

      val jsiRuntimePointer: Long =
        catalystInstance
          .javaScriptContextHolder
          .get()
          .toLong()

      if (jsiRuntimePointer != 0L) {
        installJSIBindings(jsiRuntimePointer)
      }
    }

    Constant("PI") {
      Math.PI
    }

    Events("onChange")

    Function("hello") {
      "Hello world! 👋"
    }

    AsyncFunction("setValueAsync") { value: String ->
      sendEvent(
        "onChange",
        mapOf(
          "value" to value
        )
      )
    }

    View(ExpoImageToSvgView::class) {

      Prop("url") { view: ExpoImageToSvgView, url: URL ->
        view.webView.loadUrl(url.toString())
      }

      Events("onLoad")
    }
  }

  private external fun installJSIBindings(
    jsiRuntimePointer: Long
  )
}