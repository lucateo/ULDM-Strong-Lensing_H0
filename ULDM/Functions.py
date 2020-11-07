import csv
from lenstronomy.Cosmo.lens_cosmo import LensCosmo

# Various side functions used in main program
############# SHOULD ADD M SOLITON ONCE YOU HAVE IT
def properties(name, file_name, partial_name=False):
    """! Function to extract parameters from pre-computed csv file
    @param name Name of the galaxy
    @param file_name The name of the file from which you take data
    @partial_name Bool
    @return Values of Einstein angle, effective radius, \f$ \gamma_{\text{pl}} \f$, redshift of lens and source, velocity dispersion

    """
    with open(file_name, newline='') as myFile:
        reader = csv.DictReader(myFile)
        for row in reader:
            if (partial_name and name[:4] == str(row['name'])[:4]) or name == str(row['name']):

            #if name == str(row['name']):
                theta_E = float(row['theta_E'])
                z_lens = float(row['z_lens'])
                z_source = float(row['z_source'])
                r_eff = float(row['r_eff'])
                lensCosmo = LensCosmo(z_lens, z_source, cosmo=None)
                sigma_sis = lensCosmo.sis_theta_E2sigma_v(theta_E)
                try:
                    gamma_pl = float(row['gamma'])
                except:
                    gamma_pl = -1
                return theta_E, r_eff, gamma_pl, z_lens, z_source, sigma_sis

