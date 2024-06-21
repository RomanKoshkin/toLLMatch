import type {URLParamNames} from './types/URLParamsTypes';

export function getBooleanParamFlag(
  flag: URLParamNames,
  defaultValue?: boolean,
): boolean {
  const paramFlagValue = getBooleanParamFlagWithoutDefault(flag);

  if (paramFlagValue == null) {
    // The default value for paramFlags is false, unless they explicitly provide a
    // defaultValue via the config
    return defaultValue ?? false;
  }

  return paramFlagValue;
}

export function getBooleanParamFlagWithoutDefault(
  flag: URLParamNames,
): boolean | null {
  const urlParams = new URLSearchParams(window.location.search);

  if (urlParams.get(flag) == null) {
    return null;
  }

  return urlParams.get(flag) !== '0';
}

export function getStringParamFlag(
  flag: URLParamNames,
  defaultValue?: string,
): string | null {
  const urlParams = new URLSearchParams(window.location.search);

  const param = urlParams.get(flag);

  return param ?? defaultValue ?? null;
}
