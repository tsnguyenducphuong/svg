import { requireNativeView } from 'expo';
import * as React from 'react';

import { ExpoImageToSvgViewProps } from './ExpoImageToSvg.types';

const NativeView: React.ComponentType<ExpoImageToSvgViewProps> =
  requireNativeView('ExpoImageToSvg');

export default function ExpoImageToSvgView(props: ExpoImageToSvgViewProps) {
  return <NativeView {...props} />;
}
