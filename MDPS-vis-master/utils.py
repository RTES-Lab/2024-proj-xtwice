import yaml
from box import Box

def load_yaml(config_yaml_file: str):
    """
    YAML 파일을 읽어와 Box 객체로 변환하는 함수.

    Parameters
    ----------
    config_yaml_file : str
        읽을 YAML 파일의 경로.

    Returns
    ----------
    config : Box
        YAML 파일의 내용을 포함한 Box 객체
    """
    with open(config_yaml_file) as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        config = Box(config_yaml)
    return config