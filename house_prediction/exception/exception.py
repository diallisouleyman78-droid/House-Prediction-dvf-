import sys
from house_prediction.logging.logger import logging

class HousePredictionException(Exception):
    def __init__(self, error_message, error_detail: sys):
        self.error_message = error_message
        _, _, exc_tb = error_detail.exc_info()
        
        self.line_number = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename
    
    def __str__(self):
        return f"Error occurred in script: {self.file_name} at line number: {self.line_number} error message: {self.error_message}"


# Test the exception
if __name__ == "__main__":
    try:
        # Simulate an error
        raise HousePredictionException("This is a test error message", sys)
    except HousePredictionException as e:
        print(e)
        print("\nException test successful!")