import type { StyleProp, ViewStyle } from 'react-native';

export type OnLoadEventPayload = {
  url: string;
};

export type ExpoImageToSvgModuleEvents = {
  onChange: (params: ChangeEventPayload) => void;
};

export type ChangeEventPayload = {
  value: string;
};

export type ExpoImageToSvgViewProps = {
  url: string;
  onLoad: (event: { nativeEvent: OnLoadEventPayload }) => void;
  style?: StyleProp<ViewStyle>;
};

 
export interface VectorizeOptions {
  /** * The pixel buffer (Uint8Array). 
   * Note: The JSI layer expects the .buffer property of this array.
   */
  buffer: Uint8Array;
  width: number;
  height: number;
  /** * Color precision (1-8). Higher is more detailed. 
   * Default: 6
   */
  precision?: number;
  /** * Turning angle (degrees) to detect hard corners. 
   * Default: 60
   */
  threshold?: number;
  /** * Discard clusters smaller than this pixel count. 
   * Default: 4
   */
  filterSpeckle?: number;
}