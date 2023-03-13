try:

    from enum import Enum
    from io import BytesIO, StringIO
    from typing import Union

    import pandas as pd
    import streamlit as st
except Exception as e:
    print(e)

STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""


class FileUpload(object):

    def __init__(self):
        self.fileTypes = ["csv", "xlsx", "txt", "json"]

    def run(self):
        """
        Upload File on Streamlit Code
        :return:
        """
        st.info(__doc__)
        st.markdown(STYLE, unsafe_allow_html=True)
        file = st.file_uploader("Upload file", type=self.fileTypes)
        show_file = st.empty()
        if not file:
            show_file.info("Please upload a file of type: " +
                           ", ".join(["csv", "xlsx", "txt", "json"]))
            return
        ext = file.name.split(".")[-1]
        if file is not None:
            if ext == "csv":
                data = pd.read_csv(file)
                st.dataframe(data.head())
            elif ext == "xlsx":
                data = pd.read_excel(file)
                st.dataframe(data.head())
            else:
                st.write("Error: upload the file in csv or excel format")
        else:
            return None

        file.close()


if __name__ == "__main__":
    helper = FileUpload()
    helper.run()
