export default function setURLParam<T>(
  paramName: string,
  value: T,
  // If there's no defaultValue specified then we always set the URL param explicitly
  defaultValue?: T,
): void {
  const urlParams = new URLSearchParams(window.location.search);
  if (defaultValue != null && value === defaultValue) {
    urlParams.delete(paramName);
  } else {
    let stringValue: string;

    switch (typeof value) {
      case 'string':
        stringValue = value;
        break;
      case 'boolean':
        stringValue = value ? '1' : '0';
        break;
      default:
        throw new Error(`Unsupported URL param type: ${typeof value}`);
    }

    if (urlParams.has(paramName)) {
      urlParams.set(paramName, stringValue);
    } else {
      urlParams.append(paramName, stringValue);
    }
  }

  const paramStringWithoutQuestionMark = urlParams.toString();

  window.history.replaceState(
    null,
    '',
    `${window.location.pathname}${
      paramStringWithoutQuestionMark.length > 0 ? '?' : ''
    }${paramStringWithoutQuestionMark}`,
  );
}
