import copy

import numpy as np


class Thorus(np.ndarray):

    @staticmethod
    def make(array):
        array = np.asarray(array)
        T = Thorus(array.shape, dtype=array.dtype)
        T[:] = array[:]
        return T

    def __getitem__(self, key):
        if np.issubdtype(type(key), np.integer):
            if key >= self.shape[0]:
                key_new = key % self.shape[0]
            else:
                key_new = key
            return np.asarray(super().__getitem__(key_new))

        elif isinstance(key, slice):
            key_new = slice(0 if key.start is None else key.start, self.shape[0] if key.stop is None else key.stop,
                            1 if key.step is None else key.step)

            if key_new.stop <= self.shape[0] and key_new.start >= 0:
                return np.asarray(super().__getitem__(key_new))

            elif key_new.stop > self.shape[0] and key_new.start >= self.shape[0] and key_new.stop - key_new.start < \
                    self.shape[0] and key_new.stop - key_new.start > 0:
                key_new = slice(key_new.start % self.shape[0], key_new.stop % self.shape[0], key_new.step)

            elif key_new.stop > self.shape[0] and key_new.start < self.shape[0]:
                new_key_one = slice(key_new.start % self.shape[0], self.shape[0], key_new.step)
                new_key_two = slice(0, key_new.stop % self.shape[0], key_new.step)
                return np.concatenate([super().__getitem__(new_key_one), super().__getitem__(new_key_two)], axis=0)

            elif key_new.stop == 0 and key_new.start < 0:
                key_new = slice(key_new.start % self.shape[0], self.shape[0], key_new.step)

            elif key_new.stop > 0 and key_new.stop < self.shape[0] and key_new.start < 0:
                new_key_one = slice(key_new.start % self.shape[0], self.shape[0], key_new.step)
                new_key_two = slice(0, key_new.stop % self.shape[0], key_new.step)
                return np.concatenate([super().__getitem__(new_key_one), super().__getitem__(new_key_two)], axis=0)

            elif key_new.stop < 0 and key_new.start < 0:
                key_new = key

            elif key_new.stop >= self.shape[0] and key_new.start >= self.shape[0]:
                key_new = key

            return np.asarray(super().__getitem__(key_new))

        elif isinstance(key, tuple):
            if len(key) > 0:
                return self._get_item_split(list(key), 0)
            else:
                return super().__getitem__(key)
        else:
            raise ValueError('The type is key is confusing {}'.format(key))

    def _get_item_split(self, key, level):
        if level == len(key):
            key = tuple(key)
            return np.asarray(super().__getitem__(key))

        ck = key[level]

        if np.issubdtype(type(ck), np.integer):
            if ck >= self.data.shape[level]:
                ck = ck % self.data.shape[level]
            key[level] = ck
            return self._get_item_split(key, level + 1)

        elif isinstance(ck, slice):
            ck = slice(0 if ck.start is None else ck.start, self.data.shape[level] if ck.stop is None else ck.stop,
                       1 if ck.step is None else ck.step)

            if ck.stop <= self.data.shape[level] and ck.start >= 0:
                return self._get_item_split(key, level + 1)

            elif ck.stop >= self.data.shape[level] and ck.start >= self.data.shape[level] and ck.stop - ck.start < \
                    self.data.shape[level]:
                ck = slice(ck.start % self.data.shape[level], ck.stop % self.data.shape[level], ck.step)
                key[level] = ck
                return self._get_item_split(key, level + 1)

            elif ck.stop >= self.data.shape[level] and ck.start < self.data.shape[level]:
                new_sl_one = slice(ck.start, self.data.shape[level], ck.step)
                new_k_one = copy.deepcopy(key)
                new_k_one[level] = new_sl_one

                new_sl_two = slice(0, ck.stop % self.data.shape[level], ck.step)
                new_k_two = copy.deepcopy(key)
                new_k_two[level] = new_sl_two

                part_one = self._get_item_split(new_k_one, level + 1)

                part_two = self._get_item_split(new_k_two, level + 1)

                cnt_ints = np.sum([isinstance(i, int) for i in key[0:level]], dtype=np.int)
                return np.concatenate((part_one, part_two), axis=level - cnt_ints)

            elif ck.stop >= 0 and ck.start < 0:
                new_sl_one = slice(ck.start % self.data.shape[level], self.data.shape[level], ck.step)
                new_k_one = copy.deepcopy(key)
                new_k_one[level] = new_sl_one

                new_sl_two = slice(0, ck.stop, ck.step)
                new_k_two = copy.deepcopy(key)
                new_k_two[level] = new_sl_two

                part_one = self._get_item_split(new_k_one, level + 1)

                part_two = self._get_item_split(new_k_two, level + 1)

                cnt_ints = np.sum([isinstance(i, int) for i in key[0:level]], dtype=np.int)
                return np.concatenate((part_one, part_two), axis=level - cnt_ints)

            elif ck.stop < 0 and ck.start < 0:
                return self._get_item_split(key, level + 1)

            elif ck.stop >= self.data.shape[level] and ck.start >= self.data.shape[level]:
                return self._get_item_split(key, level + 1)

            else:
                return self._get_item_split(key, level + 1)

        else:
            raise ValueError('The type is key is confusing {}'.format(key))

    def __setitem__(self, key, value):
        if np.issubdtype(type(key), np.integer):
            if key >= self.shape[0] or key < 0:
                key_new = key % self.shape[0]
            else:
                key_new = key
            super().__setitem__(key_new, value)

        elif isinstance(key, slice):
            key_new = slice(0 if key.start is None else key.start, self.shape[0] if key.stop is None else key.stop,
                            1 if key.step is None else key.step)

            if key_new.stop <= self.shape[0] and key_new.start >= 0:
                super().__setitem__(key_new, value)

            elif key_new.stop >= self.shape[0] and key_new.start >= self.shape[0] and key_new.stop - key_new.start < \
                    self.shape[0]:
                key_new = slice(key_new.start % self.shape[0], key_new.stop % self.shape[0], key_new.step)
                super().__setitem__(key_new, value)

            elif key_new.stop >= self.shape[0] and key_new.start < self.shape[0]:
                new_key_new = slice(key_new.start, self.shape[0], key_new.step)
                splitter = self.shape[0] - key_new.start
                super().__setitem__(new_key_new, value[:splitter])

                new_key_new = slice(0, key_new.stop % self.shape[0], key_new.step)
                super().__setitem__(new_key_new, value[splitter:])

            elif key_new.stop >= 0 and key_new.start < 0:
                new_key_new = slice(key_new.start + self.shape[0], self.shape[0], key_new.step)

                splitter = self.shape[0] + key_new.start
                super().__setitem__(new_key_new, value[:splitter])

                new_key_new = slice(0, key_new.stop, key_new.step)
                super().__setitem__(new_key_new, value[splitter:])

            elif key_new.stop < 0 and key_new.start < 0:
                key_new = slice(key_new.start % self.shape[0], key_new.stop % self.shape[0], key_new.step)
                super().__setitem__(key_new, value)

            elif key_new.stop >= self.shape[0] and key_new.start >= self.shape[0]:
                super().__setitem__(key_new, value)

            else:
                super().__setitem__(key, value)

        elif isinstance(key, tuple):
            if len(key) > 0:
                self._set_item_split(list(key), value, 0)
            else:
                super().__setitem__(key, value)
        else:
            raise ValueError('The type is key is confusing {}'.format(key))

    def _set_item_split(self, key, value, level):
        if level == len(key):
            key = tuple(key)
            super().__setitem__(key, value)
        else:
            ck = key[level]
            if np.issubdtype(type(ck), np.integer):
                if ck >= self.data.shape[level]:
                    ck = ck % self.data.shape[level]
                key[level] = ck

                self._set_item_split(key, value, level + 1)

            elif isinstance(ck, slice):

                ck = slice(0 if ck.start is None else ck.start, self.data.shape[level] if ck.stop is None else ck.stop,
                           1 if ck.step is None else ck.step)

                if ck.stop <= self.data.shape[level] and ck.start >= 0:
                    self._set_item_split(key, value, level + 1)

                elif ck.stop >= self.data.shape[level] and ck.start >= self.data.shape[level] and ck.stop - ck.start < \
                        self.data.shape[level]:
                    ck = slice(ck.start % self.data.shape[level], ck.stop % self.data.shape[level], ck.step)
                    key[level] = ck
                    self._set_item_split(key, value, level + 1)

                elif ck.stop >= self.data.shape[level] and ck.start < self.data.shape[level]:
                    cnt_ints = np.sum([isinstance(i, int) for i in key[0:level]], dtype=np.int)
                    new_sl_one = slice(ck.start, self.data.shape[level], ck.step)

                    new_k_one = copy.deepcopy(key)
                    new_k_one[level] = new_sl_one
                    splitter = self.data.shape[level] - ck.start
                    value_one, value_two = np.split(value, [splitter], axis=level - cnt_ints)

                    self._set_item_split(new_k_one, value_one, level + 1)

                    new_sl_two = slice(0, ck.stop % self.data.shape[level], ck.step)
                    new_k_two = copy.deepcopy(key)
                    new_k_two[level] = new_sl_two
                    self._set_item_split(new_k_two, value_two, level + 1)

                elif ck.stop >= 0 and ck.start < 0:
                    cnt_ints = np.sum([isinstance(i, int) for i in key[0:level]], dtype=np.int)

                    new_sl_one = slice(ck.start % self.data.shape[level], self.data.shape[level], ck.step)
                    new_k_one = copy.deepcopy(key)
                    new_k_one[level] = new_sl_one

                    splitter = self.data.shape[level] - (ck.start % self.data.shape[level])
                    value_one, value_two = np.split(value, [splitter], axis=level - cnt_ints)

                    self._set_item_split(new_k_one, value_one, level + 1)

                    new_sl_two = slice(0, ck.stop, ck.step)
                    new_k_two = copy.deepcopy(key)
                    new_k_two[level] = new_sl_two
                    self._set_item_split(new_k_two, value_two, level + 1)

                elif ck.stop < 0 and ck.start < 0:
                    self._set_item_split(key, value, level + 1)

                elif ck.stop >= self.data.shape[level] and ck.start >= self.data.shape[level]:
                    self._set_item_split(key, value, level + 1)

                else:
                    self._set_item_split(key, value, level + 1)

            else:
                raise ValueError('The type is key is confusing {}'.format(key))
