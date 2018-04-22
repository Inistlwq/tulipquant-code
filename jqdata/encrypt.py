from Crypto.Cipher import AES
import sys
import base64
import binascii
if sys.version_info[0] == 3:
    xrange = range



def pad_text16(text):
    from six import StringIO
    l = len(text)
    output = StringIO()
    val = 16 - (l % 16)
    for _ in xrange(val):
        output.write('%02x' % val)
    bin_str = binascii.unhexlify(output.getvalue())
    if sys.version_info[0] == 3:
        return text + bin_str.decode()
    else:
        return text + bin_str


def unpad_text16(text):
    nl = len(text)
    val = int(binascii.hexlify(text[-1]), 16)
    if val > 16:
        raise ValueError('Input is not padded or padding is corrupt')
    l = nl - val
    return text[:l]

def encryptByKey(key, orgtext, iv):
    encryptor = AES.new(key, AES.MODE_CBC, iv)
    result = encryptor.encrypt(pad_text16(orgtext))
    return base64.b64encode(result)

def decryptByKey(key, orgtext, iv):
    orgtext = orgtext.replace(' ', '+')
    orgtext = base64.b64decode(orgtext)
    decryptor = AES.new(key, AES.MODE_CBC, iv)
    result = decryptor.decrypt(orgtext)
    return result.rstrip('\0')
    
if __name__ == "__main__":
    iv = "1234567890123456"
    orgtext = "select * from stock_valuation limit 2"
    encrypted = encryptByKey(iv, orgtext, iv)
    print ('#####encrypted: ', encrypted)
    print ('#####decrypted: ', unpad_text16(decryptByKey(iv, encrypted, iv)))

    sql = 'O70ldyARkho18mYP4Omb09i6lv+o/lsU1NsmIGH2I8apDhgEXWYwK+T2DF9dCx4X'
    print('#:', decryptByKey(iv, sql, iv))