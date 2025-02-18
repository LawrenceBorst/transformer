import os
from requests import Response
import requests


class CorpusDownloader:
    """
    A class that lets you download a dataset and convert it to a corpus .txt file

    Args:
        dataset_url (str): the url to the dataset
        output_path (str): the output path to save the downloaded dataset
        avoid_overwrite (bool): whether to avoid overwriting an existing file
    """

    _dataset_url: str
    _output_path: str
    _avoid_overwrite: bool

    def __init__(
        self,
        dataset_url: str,
        output_path: str,
        avoid_overwrite: bool = False,
    ) -> None:
        self._dataset_url = dataset_url
        self._output_path = output_path
        self._avoid_overwrite = avoid_overwrite

    def save(self) -> None:
        if self._avoid_overwrite and os.path.exists(self._output_path):
            return

        os.makedirs(
            os.path.dirname(self._output_path),
            exist_ok=True,
        )

        response: Response = requests.get(self._dataset_url, stream=True)

        if response.status_code == 200:
            with open(self._output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    """
                    No sort of cleaning is wanted here, as we expect the model to take
                    in data in the form of the dataset, to be autocompleted that is
                    """
                    f.write(chunk)
                print(f"File downloaded successfully: {self._output_path}")
        else:
            print(f"Could not download file. Status code: {response.status_code}")
