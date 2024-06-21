import {random} from 'lodash';

// const USABLE_CHARACTERS = 'BCDFGHJKMPQRTVWXY2346789';
const USABLE_CHARACTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
const ID_LENGTH = 4;

export function isValidRoomID(id: string | null | undefined): boolean {
  if (id == null) {
    return false;
  }
  if (id.length !== ID_LENGTH) {
    return false;
  }
  return isValidPartialRoomID(id);
}

export function isValidPartialRoomID(roomID: string): boolean {
  return (
    roomID.length <= ID_LENGTH &&
    roomID.split('').every((char) => USABLE_CHARACTERS.includes(char))
  );
}

export default function generateNewRoomID(): string {
  return Array.from(
    {length: ID_LENGTH},
    () => USABLE_CHARACTERS[random(USABLE_CHARACTERS.length - 1)],
  ).join('');
}

export function getSequentialRoomIDForTestingGenerator(): () => string {
  let counter = 0;

  return function generateNextRoomID(): string {
    const counterInBase: string = Number(counter)
      .toString(USABLE_CHARACTERS.length)
      .padStart(ID_LENGTH, '0');

    if (counterInBase.length > ID_LENGTH) {
      throw new Error(
        'Ran out of unique room IDs from the sequential generator',
      );
    }

    const result = counterInBase
      .split('')
      .map(
        (digit) => USABLE_CHARACTERS[parseInt(digit, USABLE_CHARACTERS.length)],
      )
      .join('');

    counter++;

    return result;
  };
}
