import configparser
import argparse


def show_config(ini):
    '''
    設定ファイルの全ての内容を表示する（コメントを除く）
    '''
    for section in ini.sections():
        print("[" + section + "]")
        show_section(ini, section)
    return


def show_section(ini, section):
    '''
    設定ファイルの特定のセクションの内容を表示する
    '''
    for key in ini.options(section):
        show_key(ini, section, key)
    return


def show_key(ini, section, key):
    '''
    設定ファイルの特定セクションの特定のキー項目（プロパティ）の内容を表示する
    '''
    print(section + "." + key + " = " + ini.get(section, key))
    return


def read_config():
    # パラメータの読み込み
    parser = argparse.ArgumentParser(description='パラメータファイルの読み込み')
    parser.add_argument('-config', required=True)
    parser.add_argument('--debug',
                        action='store_true',
                        default=False,
                        help='debug mode if this flag is set (default: False)')
    args = parser.parse_args()
    INI_FILE = args.config
    print("Config file:", INI_FILE)
    ini = configparser.SafeConfigParser()
    ini.read(INI_FILE, encoding='utf8')
    show_config(ini)
    return ini, args.debug


def read_config_for_pred():
    # パラメータの読み込み
    parser = argparse.ArgumentParser(description='パラメータファイルの読み込み')
    parser.add_argument('-config', required=True)
    parser.add_argument('-file', required=True)
    parser.add_argument('-dir', required=True)
    parser.add_argument('--debug',
                        action='store_true',
                        default=False,
                        help='debug mode if this flag is set (default: False)')
    args = parser.parse_args()
    INI_FILE = args.config
    print("Config file:", INI_FILE)
    ini = configparser.SafeConfigParser()
    ini.read(INI_FILE, encoding='utf8')
    show_config(ini)
    return ini, args.debug, args.file, args.dir
