import locale
import pandas as pd
import holidays
# from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess


class TimeSeriesFeatureEngineering:

    def feature_engineering_time_series_dynamic(self, time_series_data: pd.DataFrame, remove_na=True):
        """
        Perform dynamic feature engineering on time series, adding columns as applicable.

        Parameters:
        - time_series_data: pandas DataFrame or Series. The original time series with a valid time index.
        - remove_na: bool, optional. Indicates whether to remove rows with missing values after feature engineering.

        Returns:
        - DataFrame: A new DataFrame containing the generated features.
        - date_attributes: list of str. The date-related attributes added to the DataFrame.
        """
        if not isinstance(time_series_data, pd.DataFrame):
            raise ValueError("time_series_data must be a pandas DataFrame")

        self.has_datetime_index(time_series_data)

        # Make a copy of the original data to avoid modifying the input DataFrame
        data = time_series_data.copy()


        # Convert the index to a DatetimeIndex if needed
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # Add date-related features if the index is of type DatetimeIndex
        if isinstance(data.index, pd.DatetimeIndex):
            # Get all available date attributes
            # date_attributes = ['year', 'month',
            #                    'day', 'day_of_week', 'day_of_year']
            date_attributes = ['year', 'month']

            # Check if hour-related attributes are present
            if data.index.hour.min() > 0:
                date_attributes += ['hour', 'minute', 'second']

            # Check if week-related attributes are present
            # if hasattr(data.index, 'isocalendar'):
            #     iso_week = data.index.isocalendar().week
            #     data['isoweek'] = iso_week
            #     date_attributes += ['isoweek']

            for attr in date_attributes:
                if hasattr(data.index, attr):
                    data[attr] = getattr(data.index, attr)

        # Remove rows with missing values if necessary
        if remove_na:
            data.dropna(inplace=True)

        # Add is_first_month and is_last_month columns
        data['is_first_month'] = data.index.to_series().apply(lambda x: self.is_first_or_last_month(x, is_first=True))
        data['is_last_month'] = data.index.to_series().apply(self.is_first_or_last_month)
        data['days_in_month'] = data.index.days_in_month
        return data, date_attributes

    def has_datetime_index(self, dataframe: pd.DataFrame) -> bool:
        """
        Checks if a DataFrame has a DatetimeIndex as its index.

        Parameters:
            dataframe (pd.DataFrame): The DataFrame to be checked.

        Returns:
            bool: True if any column of the index has datetime64 type, False otherwise.
        """
        if not isinstance(dataframe.index, pd.DatetimeIndex):
            raise ValueError("The DataFrame does not have a DatetimeIndex as its index.")
        return True


    # def create_deterministic_matrix(self, data, freq="M", order=2):
    #     """
    #     Create a deterministic matrix with Fourier-based seasonal components.

    #     Parameters:
    #     - data: pandas DataFrame or Series. The original time series data with a valid time index.
    #     - freq: str, optional. The frequency of the seasonality (e.g., 'M' for monthly, 'Q' for quarterly).
    #     - order: int, optional. The number of Fourier components to include.

    #     Returns:
    #     - X: pandas DataFrame. The deterministic matrix with seasonal components.
    #     """
    #     fourier = CalendarFourier(freq=freq, order=order)
    #     dp = DeterministicProcess(
    #         index=data.index,
    #         constant=True,
    #         order=1,
    #         seasonal=True,
    #         additional_terms=[fourier],
    #         drop=True,
    #     )
    #     X = dp.in_sample()

    #     return X

    def is_first_or_last_month(self, date, is_first=False):
        """
        Determina si un mes es el primer mes o el último mes del año y devuelve un valor binario.

        Parameters:
            date (datetime): Fecha que representa el mes a evaluar.
            is_first (bool): Indica si se quiere determinar si es el primer mes (True) o el último mes (False).
                            Por defecto, es False.

        Returns:
            int: Valor binario que indica si el mes es el primero o el último. 1 representa True, 0 representa False.

        """
        month = date.month
        return int(month == 1) if is_first else int(month == 12)

    def count_holidays_in_month(self, year: int, month: int, country_code: str = 'CO') -> int:
        """
        Count the number of holidays in a specific month of a given year for a specified country.

        Parameters:
            - year (int): The year for which you want to count holidays.
            - month (int): The month for which you want to count holidays (1 to 12).
            - country_code (str): The country code for the desired country (default is 'CO' for Colombia).

        Returns:
            - int: The number of holidays in the specified month.
        """
        if month < 1 or month > 12:
            raise ValueError("Month must be in the range 1 to 12")
            # Configure the locale for the specified country
        # locale.setlocale(locale.LC_MONETARY, f'{country_code}.UTF-8')
        locale.setlocale(locale.LC_MONETARY, 'es_CO.UTF-8')

        # Create a dictionary of holidays for the specified country
        country_holidays = getattr(holidays, country_code)()

        # Get the start and end dates for the specified month
        start_date = pd.Timestamp(year, month, 1)

        end_date = pd.Timestamp(year, month, 1) + \
            pd.DateOffset(months=1) - pd.DateOffset(days=1)

        # Generate a date range for the specified month
        date_range = pd.date_range(start=start_date, end=end_date)

        # Count the number of holidays in the specified month
        num_holidays = sum(
            1 for date in date_range if country_holidays.get(date) is not None)

        return num_holidays

    def add_monthly_holiday_count(self, data_frame: pd.DataFrame, name_column: str = 'Monthly_Holiday_Count', country_code: str = 'CO') -> pd.DataFrame:
        """
        Add a column to the DataFrame with the count of holidays in each month.

        Parameters:
            - data_frame (pd.DataFrame): The DataFrame to which the column will be added.
            - country_code (str): The country code for the desired country (default is 'CO' for Colombia).
        """
        # Initialize an empty list to store the holiday counts
        holiday_counts = []

        # Iterate through the index of the DataFrame to get the year and month for each date
        for date in data_frame.index:
            year = date.year
            month = date.month

            # Count the holidays for the specific month and year

            num_holidays = self.count_holidays_in_month(
                year, month, country_code)

            # Append the count to the list
            holiday_counts.append(num_holidays)

        # Add a new column 'Monthly_Holiday_Count' to the DataFrame
        data_frame[name_column] = holiday_counts
        return data_frame
