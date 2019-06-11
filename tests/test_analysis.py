import unittest
import numpy as np
import analysis


class TestAnalysis(unittest.TestCase):
    def test_widim_settings_init_WS(self):
        """
        Checks that the initial window size must be
        odd,
        5 <= init_WS <= 245  (245, because adding 10 takes it to 255, which is
                              is the biggest fft (i.e. 256) I want to allow)
        Must be >= than the final WS
        """

        # check that a Value error is raised if init_WS is even
        with self.assertRaises(ValueError):
            analysis.widim_settings(init_WS=96)

        # check that a Value error is raised if init_WS is less than 5
        # set final WS to 3 too to prevent init_WS < final_WS being violated
        with self.assertRaises(ValueError):
            analysis.widim_settings(init_WS=3, final_WS=3)

        # check that a value error is not raised if init_WS is 5
        # set final WS to 5 too to prevent init_WS < final_WS being violated
        analysis.widim_settings(init_WS=5, final_WS=5)

        # check that a Value error is raised if init_WS is greater than 245
        with self.assertRaises(ValueError):
            analysis.widim_settings(init_WS=257)

        # check that a value error is not raised if init_WS is 245

        analysis.widim_settings(init_WS=245)

        # check that a Value error is raised if init_WS is less than final_WS
        with self.assertRaises(ValueError):
            analysis.widim_settings(init_WS=97, final_WS=101)

        # check that a value error is not raised if init_WS == final_WS
        analysis.widim_settings(init_WS=97, final_WS=97)

    def test_widim_settings_final_WS(self):
        """
        Checks that the final window size must be
        odd,
        5 <= final_WS <= 245  (245, because the adding 10 takes it to 255, which is
                              is the biggest fft (i.e. 256) I want to allow)
        Must be <= than the initial WS
        """

        # check that a Value error is raised if final_WS is even
        with self.assertRaises(ValueError):
            analysis.widim_settings(final_WS=32)

        # check that a Value error is raised if final_WS is less than 5
        with self.assertRaises(ValueError):
            analysis.widim_settings(final_WS=3)

        # check that a value error is not raised if final_WS is 5
        analysis.widim_settings(final_WS=5)

        # check that a Value error is raised if final_WS is greater than 245
        with self.assertRaises(ValueError):
            analysis.widim_settings(final_WS=257)

        # check that a value error is not raised if final_WS is 245
        # init_ws to 245 to satisfy init_WS >= final_WS
        analysis.widim_settings(final_WS=245, init_WS=245)

    def test_widim_settings_WOR(self):
        """
        Checks that the window overlap ratio must be
        0 <= WOR < 1
        """

        # check that a value error is raised for WOR < 0
        with self.assertRaises(ValueError):
            analysis.widim_settings(WOR=-0.3)

        # check that a value error is not raised for WOR == 0
        analysis.widim_settings(WOR=0)

        # check that a value error is raised for WOR == 1
        with self.assertRaises(ValueError):
            analysis.widim_settings(WOR=1)

    def test_widim_settings_n_iter_main(self):
        """
        Checks that the number of main iterations is an integer and:
        1 <= n_iter_main <= 10
        """

        # check that a value error is raised for n_iter_main == 0
        with self.assertRaises(ValueError):
            analysis.widim_settings(n_iter_main=0)

        # check that a value error is not raised for n_iter_main == 1
        analysis.widim_settings(n_iter_main=1)

        # check that a value error is raised for n_iter_main == 11
        with self.assertRaises(ValueError):
            analysis.widim_settings(n_iter_main=11)

        # check that a value error is not raised for n_iter_main == 10
        analysis.widim_settings(n_iter_main=10)

        # check that a value error is raised if the input is not an integer
        with self.assertRaises(ValueError):
            analysis.widim_settings(n_iter_main=3.5)

    def test_widim_settings_n_iter_ref(self):
        """
        Checks that the number of refinement iterations is an integer and:
        0 <= n_iter_ref <= 10
        """

        # check that a value error is raised for n_iter_ref == -1
        with self.assertRaises(ValueError):
            analysis.widim_settings(n_iter_ref=-1)

        # check that a value error is not raised for n_iter_ref == 0
        analysis.widim_settings(n_iter_ref=0)

        # check that a value error is raised for n_iter_ref == 11
        with self.assertRaises(ValueError):
            analysis.widim_settings(n_iter_ref=11)

        # check that a value error is not raised for n_iter_ref == 10
        analysis.widim_settings(n_iter_ref=10)

        # check that a value error is raised if the input is not an integer
        with self.assertRaises(ValueError):
            analysis.widim_settings(n_iter_ref=3.5)

    def test_widim_settings_vec_val(self):
        """
        Checks that the vector validation method is one of the valid options
        for now this is only NMT
        """

        # check that a value error is raised for vec_val != 'NMT'
        with self.assertRaises(ValueError):
            analysis.widim_settings(vec_val='testing')

        # check that a value error is not raised for vec_val == 'NMT'
        analysis.widim_settings(vec_val='NMT')

    def test_widim_settings_interp(self):
        """
        Checks that the interpolation method is one of the valid options
        so far: 'struc_lin' and 'struc_cub'
        """

        # check that a value error is raised for invalid interp
        with self.assertRaises(ValueError):
            analysis.widim_settings(interp='testing')

        # check that a value error is not raised for valid interp method
        options = ['struc_lin', 'struc_cub']
        for option in options:
            analysis.widim_settings(interp=option)

    def test_widim_settings_default_config(self):
        """
        Check the default configuration of widim_settings returns the expected
        dict values
        """
        # expected
        settings = {
            "init_WS": 97,
            "final_WS": 33,
            "WOR": 0.5,
            "n_iter_main": 3,
            "n_iter_ref": 2,
            "vec_val": 'NMT',
            "interp": 'struc_cub',
        }

        self.assertEqual(settings, analysis.widim_settings())

    def test_widim_settings_with_non_default_config(self):
        """
        Check the default configuration of widim_settings returns the expected
        dict values
        """
        # expected
        settings = {
            "init_WS": 55,
            "final_WS": 15,
            "WOR": 0.75,
            "n_iter_main": 4,
            "n_iter_ref": 3,
            "vec_val": 'NMT',
            "interp": 'struc_lin',
        }

        self.assertEqual(
            settings, analysis.widim_settings(init_WS=55,
                                              final_WS=15,
                                              WOR=0.75,
                                              n_iter_main=4,
                                              n_iter_ref=3,
                                              interp='struc_lin'))

    def test_calculate_WS_specific_inputs(self):
        """
        This is testing that the 'special' cases are handle correctly.
        iter_ = 1 returns ['init_WS']
            UNLESS
            iter_ = 1 and ['n_iter_main'] == 1, which returns ['final_WS']
        iter_ >= ['n_iter_main'] returns ['final_WS']
        """

        # create settings dict
        settings = analysis.widim_settings(n_iter_main=4, n_iter_ref=2,
                                           init_WS=57, final_WS=33)

        # input with iter > n_iter_main => WS = final_WS
        WS = analysis.WS_for_iter(6, settings)
        self.assertEqual(WS, 33)

        # input with iter == n_iter_main => WS = final_WS
        WS = analysis.WS_for_iter(4, settings)
        self.assertEqual(WS, 33)

        # input with iter == 1 => WS = init_WS
        WS = analysis.WS_for_iter(1, settings)
        self.assertEqual(WS, 57)

        # input with iter == 1, n_iter_main == 1 => WS = init_WS
        settings['n_iter_main'] = 1
        WS = analysis.WS_for_iter(1, settings)
        self.assertEqual(WS, 33)

    def test_calculate_WS_middle_input(self):
        """
        Simply tests an example usage
        """

        settings = analysis.widim_settings(n_iter_main=3,
                                           init_WS=97,
                                           final_WS=25)

        exp = 49
        self.assertEqual(analysis.WS_for_iter(2, settings), exp)


if __name__ == "__main__":
    unittest.main(buffer=True)
