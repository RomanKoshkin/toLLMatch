import { getBooleanParamFlag, getStringParamFlag } from './getParamFlag';
import { URLParamsObject } from './types/URLParamsTypes';

/**
 * These are the URL parameters you can provide to the app to change its behavior.
 *
 * Boolean flags can be set by just providing the flag name (`?autoJoin`), or by
 * explicitly setting it to 1 (true) or 0 (false): `?autoJoin=1` or `?autoJoin=0`
 *
 * String flags require an explicit value: `?roomID=ABCD`
 *
 * Examples:
 *
 * - `http://localhost:5173/?roomID=BBCD&autoJoin&debug`
 * - `http://localhost:5173/?serverURL=localhost:8000`

 * @returns
 */

export function getURLParams(): URLParamsObject {
  return {
    // animate the translation text when it arrives, typing it out one letter at a time
    animateTextDisplay: getBooleanParamFlag('animateTextDisplay', true), // default to true;

    // automatically join the room when the app loads. requires roomID to be set via url param as well
    autoJoin: getBooleanParamFlag('autoJoin', false),

    // automatically check the server debug flag as true
    debug: getBooleanParamFlag('debug', false),

    // Enable UI on the client that allows locking out other users of the server when it's being used for high profile demos
    // NOTE: There is an escape hatch for disabling a server lock by setting the name field to remove_server_lock
    enableServerLock: getBooleanParamFlag('enableServerLock', false),

    // Pre-populate the Room Code field with the provided roomID. Can be used in conjunction with autoJoin to jump straight into the room
    roomID: getStringParamFlag('roomID'),

    // Use an alternate server URL as the streaming server (useful for pointing to dev servers: http://localhost:5173/?serverURL=localhost:8000)
    serverURL: getStringParamFlag('serverURL'),

    // Skip the popup dialog that displays within VR, which is mostly redundant with the web based dialog
    skipARIntro: getBooleanParamFlag('skipARIntro', true), // default to true

    // Shows the translation text in AR in front of an opaque panel covering all the text area
    // single_block = original single text block with background
    // lines = each line is a separate block and animates
    // lines_with_background = adds a panel behind lines
    ARTranscriptionType: getStringParamFlag('ARTranscriptionType') || 'lines',
  };
}
