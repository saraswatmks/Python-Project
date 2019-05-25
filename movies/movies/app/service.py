from movies.data.get_data import CreateData


class FinalShot:

    @staticmethod
    def run():
        cls = CreateData(a=10, b=20)
        cf = cls.get_data()
        print(f'this is cf: {cf}')
        df = cls.get_df()
        print(f'this is df: {df}')
