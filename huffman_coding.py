import io

class HuffmanTable:
    def __init__(self, bits, huffval):
        
        self.bits = list(bits)
        self.huffval = list(huffval)
        self.encode_table = {}
        self.decode_table = {}
        self.max_code_len = 0

        self._generate_huffman_codes()
        self._build_decode_table()

    def _generate_huffman_codes(self):
        code = 0
        si = 1
        huffval_idx = 0
        for i in range(16):
            num_codes = self.bits[i]
            for _ in range(num_codes):
                symbol = self.huffval[huffval_idx]
                self.encode_table[symbol] = (code, si)
                code += 1
                huffval_idx += 1
            code <<= 1
            si += 1
            if num_codes > 0:
                self.max_code_len = i + 1

    def _build_decode_table(self):
        self.decode_table = {}
        for symbol, (code, length) in self.encode_table.items():
            code_str = format(code, f'0{length}b')
            self.decode_table[code_str] = symbol

    def get_code(self, symbol):
        return self.encode_table.get(symbol)

    def decode_symbol(self, bit_reader):
        current_code = ""
        for _ in range(self.max_code_len):
            bit = bit_reader.read_bit()
            if bit is None:
                return None
            current_code += str(bit)
            if current_code in self.decode_table:
                return self.decode_table[current_code]
        return None

class BitWriter:
    def __init__(self):
        self._buffer = 0
        self._bit_count = 0
        self._byte_stream = bytearray()

    def write_bit(self, bit):
        if bit not in (0, 1):
            raise ValueError("Bit must be 0 or 1")
        self._buffer = (self._buffer << 1) | bit
        self._bit_count += 1
        if self._bit_count == 8:
            self._flush_byte()

    def write_bits(self, value, num_bits):
        if num_bits == 0:
            return
        mask = (1 << num_bits) - 1
        bits_to_write = value & mask
        for i in range(num_bits - 1, -1, -1):
            bit = (bits_to_write >> i) & 1
            self.write_bit(bit)

    def _flush_byte(self):

        byte_to_write = self._buffer
        self._byte_stream.append(byte_to_write)
        if byte_to_write == 0xFF:
            self._byte_stream.append(0x00)
        self._buffer = 0
        self._bit_count = 0

    def get_byte_string(self):
        if self._bit_count > 0:
            padding_bits = 8 - self._bit_count
            pad_mask = (1 << padding_bits) - 1
            self._buffer = (self._buffer << padding_bits) | pad_mask
            self._bit_count = 8
            self._flush_byte()
        return bytes(self._byte_stream)

class BitReader:
    def __init__(self, byte_data):
        self._byte_stream = io.BytesIO(byte_data)
        self._current_byte = 0
        self._bit_pos = 8
        self._marker_found = False

    def _load_byte(self):
        if self._marker_found:
            return None
        byte = self._byte_stream.read(1)
        if not byte:
            return None
        val = byte[0]
        if val == 0xFF:
            next_byte = self._byte_stream.read(1)
            if not next_byte:
                self._marker_found = True
                return None
            next_val = next_byte[0]
            if next_val == 0x00:
                self._current_byte = 0xFF
                self._bit_pos = 0
                return True
            else:
                self._byte_stream.seek(-2, 1)
                self._marker_found = True
                return None
        else:
            self._current_byte = val
            self._bit_pos = 0
            return True

    def read_bit(self):
        if self._bit_pos > 7:
            if not self._load_byte():
                return None
        bit = (self._current_byte >> (7 - self._bit_pos)) & 1
        self._bit_pos += 1
        return bit

    def read_bits(self, num_bits):

        if num_bits == 0:
            return 0
        value = 0
        for _ in range(num_bits):
            bit = self.read_bit()

            value = (value << 1) | bit
        return value

def huffman_encode_data(data_units, dc_table, ac_table):
    bit_writer = BitWriter()
    for dc_category, dc_vli_bits, ac_rle_pairs in data_units:
        dc_code_info = dc_table.get_code(dc_category)

        dc_code, dc_len = dc_code_info
        bit_writer.write_bits(dc_code, dc_len)
        if dc_category > 0:

            dc_vli_val = int(dc_vli_bits, 2)
            bit_writer.write_bits(dc_vli_val, dc_category)
        for run_length, ac_value in ac_rle_pairs:
            if run_length == 0 and ac_value == 0:
                ac_symbol = 0x00
                ac_code_info = ac_table.get_code(ac_symbol)

                ac_code, ac_len = ac_code_info
                bit_writer.write_bits(ac_code, ac_len)
                break
            elif run_length == 15 and ac_value == 0:
                ac_symbol = 0xF0
                ac_code_info = ac_table.get_code(ac_symbol)

                ac_code, ac_len = ac_code_info
                bit_writer.write_bits(ac_code, ac_len)
            else:
                from vli_coding import get_vli_category_and_value
                ac_category, ac_vli_bits = get_vli_category_and_value(ac_value)
                if ac_category == 0:
                    raise ValueError(f"Zero category for non-zero AC value {ac_value}")
                if ac_category > 15:
                    raise ValueError(f"AC VLI category {ac_category} > 15")
                if not (0 <= run_length <= 15):
                    raise ValueError(f"Invalid run_length {run_length} in AC RLE.")
                ac_symbol = (run_length << 4) | ac_category
                ac_code_info = ac_table.get_code(ac_symbol)
                if ac_code_info is None:
                    raise ValueError(f"AC symbol 0x{ac_symbol:02X} not found in Huffman table.")
                ac_code, ac_len = ac_code_info
                bit_writer.write_bits(ac_code, ac_len)
                if len(ac_vli_bits) != ac_category:
                    raise ValueError(f"Invalid VLI bits length for AC value {ac_value}")
                ac_vli_val = int(ac_vli_bits, 2)
                bit_writer.write_bits(ac_vli_val, ac_category)
    return bit_writer.get_byte_string()

def huffman_decode_data(byte_data, dc_table, ac_table, num_blocks):
    bit_reader = BitReader(byte_data)
    decoded_units = []
    from vli_coding import decode_vli
    try:
        for block_idx in range(num_blocks):
            dc_category = dc_table.decode_symbol(bit_reader)
            if dc_category is None:
                raise EOFError(f"Failed to decode DC category for block {block_idx+1}")
            dc_vli_bits = ""
            if dc_category > 0:
                dc_vli_val = bit_reader.read_bits(dc_category)
                dc_vli_bits = format(dc_vli_val, f'0{dc_category}b')
            ac_rle_pairs = []
            ac_count = 0
            while ac_count < 63:
                ac_symbol = ac_table.decode_symbol(bit_reader)
                if ac_symbol is None:
                    raise EOFError(f"Failed to decode AC symbol in block {block_idx+1} after {len(ac_rle_pairs)} pairs")
                if ac_symbol == 0x00:
                    ac_rle_pairs.append((0, 0))
                    break
                elif ac_symbol == 0xF0:
                    ac_rle_pairs.append((15, 0))
                    ac_count += 16
                else:
                    run_length = (ac_symbol >> 4) & 0x0F
                    ac_category = ac_symbol & 0x0F
                    if ac_category == 0 or ac_category > 15:
                        raise ValueError(f"Invalid AC symbol 0x{ac_symbol:02X} (run={run_length}, size={ac_category})")
                    ac_vli_val = bit_reader.read_bits(ac_category)
                    ac_vli_bits = format(ac_vli_val, f'0{ac_category}b')
                    ac_value = decode_vli(ac_category, ac_vli_bits)
                    ac_rle_pairs.append((run_length, ac_value))
                    ac_count += run_length + 1
                if ac_count > 63:
                    break
            decoded_units.append((dc_category, dc_vli_bits, ac_rle_pairs))
    except EOFError as e:
        pass
    except ValueError as e:
        pass
    return decoded_units
