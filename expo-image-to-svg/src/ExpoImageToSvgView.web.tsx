import * as React from 'react';

import { ExpoImageToSvgViewProps } from './ExpoImageToSvg.types';

export default function ExpoImageToSvgView(props: ExpoImageToSvgViewProps) {
  return (
    <div>
      <iframe
        style={{ flex: 1 }}
        src={props.url}
        onLoad={() => props.onLoad({ nativeEvent: { url: props.url } })}
      />
    </div>
  );
}
