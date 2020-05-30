from Entities.Tool import Tool
from Entities.File import File
from time import sleep
from Services.FileSystem import File_Management
from RuntimeContants import Runtime_Datasets
from Services.Configuration.Config import Config
import os
import logging


def load_tools():
    """
    Gathers all data, which is required. Takes command line args into account.
    :return:
    """

    logging.info(f"Detecting tools...")
    sleep(1)
    for file in os.listdir(Config.DATA_RAW_DIRECTORY):
        file_path = os.fsdecode(file)
        try:
            if file_path.endswith(".csv") or file_path.endswith(".tsv"):
                file_name: str = File_Management.get_file_name(file_path)
                # Remove the files version number if present, then remove the file extension to get a clean name
                tool_name: str = os.path.splitext(str(file_name.rsplit('_', 1)[0]))[0]
                tool = Tool(tool_name)

                tool_found: bool = False
                for stored_tool in Runtime_Datasets.DETECTED_TOOLS:
                    if not tool_found:
                        if stored_tool.__eq__(tool):
                            stored_tool.add_file(file_path)
                            tool_found = True
                            break

                if tool_found:
                    continue
                else:
                    if Config.VERBOSE:
                        logging.info(f"Detected tool {tool.name}")
                    tool.add_file(file_path)
                    Runtime_Datasets.DETECTED_TOOLS.append(tool)
        except OSError as ex:
            logging.warning(ex)
        except BaseException as ex:
            logging.warning(ex)

    # Verify all tools
    for tool in Runtime_Datasets.DETECTED_TOOLS:
        tool.verify()
        if Config.DEBUG_MODE:
            logging.debug(f"Tool {tool.name} is verified {tool.verified}")

    Runtime_Datasets.VERIFIED_TOOLS = [tool for tool in Runtime_Datasets.DETECTED_TOOLS if tool.verified]
    Runtime_Datasets.EXCLUDED_TOOLS = [tool for tool in Runtime_Datasets.DETECTED_TOOLS if not tool.verified]
    print()
    logging.info(
        f"Tool detector detected {len(Runtime_Datasets.VERIFIED_TOOLS)} valid tools and excluded"
        f" {len(Runtime_Datasets.EXCLUDED_TOOLS)} tools.")

    print()
    sleep(1)
