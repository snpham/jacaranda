import numpy as np
import astronomy
import bss_reader
import supercosmos




if __name__ == '__main__':
    pass

    bss_fn = 'inputs/J_MNRAS_384_775_table2.fits'
    bss_out = bss_reader.bss_import(bss_fn)
    print(bss_out)
    fn = 'inputs/SCOS_XSC_mCl1_B21.5_R20_noStepWedges.csv'
    sc_out = supercosmos.supercosmos_import(fn)
    print(sc_out)