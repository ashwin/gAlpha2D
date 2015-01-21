#include "predicates.h"

REAL gpudt_splitter;
REAL gpudt_epsilon;
REAL gpudt_resulterrbound;
REAL gpudt_ccwerrboundA, gpudt_ccwerrboundB, gpudt_ccwerrboundC;
REAL gpudt_iccerrboundA, gpudt_iccerrboundB, gpudt_iccerrboundC;
REAL gpudt_o3derrboundA, gpudt_o3derrboundB, gpudt_o3derrboundC;

//*****************************************************************************
//*                                                                           *
//*  exactinit()   Initialize the variables used for exact arithmetic.        *
//*                                                                           *
//*  `gpudt_epsilon' is the largest power of two such that 1.0 + gpudt_epsilon = 1.0 in   *
//*  floating-point arithmetic.  `gpudt_epsilon' bounds the relative roundoff       *
//*  error.  It is used for floating-point error analysis.                    *
//*                                                                           *
//*  `gpudt_splitter' is used to split floating-point numbers into two half-        *
//*  length significands for exact multiplication.                            *
//*                                                                           *
//*  I imagine that a highly optimizing compiler might be too smart for its   *
//*  own good, and somehow cause this routine to fail, if it pretends that    *
//*  floating-point arithmetic is too much like double arithmetic.              *
//*                                                                           *
//*  Don't change this routine unless you fully understand it.                *
//*                                                                           *
//*****************************************************************************
void gpudt_exactinit()
{
    REAL half;
    REAL check, lastcheck;
    int every_other;

    every_other = 1;
    half = 0.5;
    gpudt_epsilon = 1.0;
    gpudt_splitter = 1.0;
    check = 1.0;
    // Repeatedly divide `gpudt_epsilon' by two until it is too small to add to      
    //   one without causing roundoff.  (Also check if the sum is equal to     
    //   the previous sum, for machines that round up instead of using exact   
    //   rounding.  Not that these routines will work on such machines.)       
    do {
        lastcheck = check;
        gpudt_epsilon *= half;
        if (every_other) {
            gpudt_splitter *= 2.0;
        }
        every_other = !every_other;
        check = 1.0 + gpudt_epsilon;
    } while ((check != 1.0) && (check != lastcheck));
    gpudt_splitter += 1.0;
    /* Error bounds for orientation and incircle tests. */
    gpudt_resulterrbound = (3.0 + 8.0 * gpudt_epsilon) * gpudt_epsilon;
    gpudt_ccwerrboundA = (3.0 + 16.0 * gpudt_epsilon) * gpudt_epsilon;
    gpudt_ccwerrboundB = (2.0 + 12.0 * gpudt_epsilon) * gpudt_epsilon;
    gpudt_ccwerrboundC = (9.0 + 64.0 * gpudt_epsilon) * gpudt_epsilon * gpudt_epsilon;
    gpudt_iccerrboundA = (10.0 + 96.0 * gpudt_epsilon) * gpudt_epsilon;
    gpudt_iccerrboundB = (4.0 + 48.0 * gpudt_epsilon) * gpudt_epsilon;
    gpudt_iccerrboundC = (44.0 + 576.0 * gpudt_epsilon) * gpudt_epsilon * gpudt_epsilon;
    gpudt_o3derrboundA = (7.0 + 56.0 * gpudt_epsilon) * gpudt_epsilon;
    gpudt_o3derrboundB = (3.0 + 28.0 * gpudt_epsilon) * gpudt_epsilon;
    gpudt_o3derrboundC = (26.0 + 288.0 * gpudt_epsilon) * gpudt_epsilon * gpudt_epsilon;
}

//*****************************************************************************
//*                                                                           *
//*  gpudt_fast_expansion_sum_zeroelim()   Sum two expansions, eliminating zero     *
//*                                  components from the output expansion.    *
//*                                                                           *
//*  Sets h = e + f.  See my Robust Predicates paper for details.             *
//*                                                                           *
//*  If round-to-even is used (as with IEEE 754), maintains the strongly      *
//*  nonoverlapping property.  (That is, if e is strongly nonoverlapping, h   *
//*  will be also.)  Does NOT maintain the nonoverlapping or nonadjacent      *
//*  properties.                                                              *
//*                                                                           *
//*****************************************************************************

int gpudt_fast_expansion_sum_zeroelim(int elen, REAL *e, int flen, REAL *f, REAL *h)
{
    REAL Q;
    INEXACT REAL Qnew;
    INEXACT REAL hh;
    INEXACT REAL bvirt;
    REAL avirt, bround, around;
    int eindex, findex, hindex;
    REAL enow, fnow;

    enow = e[0];
    fnow = f[0];
    eindex = findex = 0;
    if ((fnow > enow) == (fnow > -enow)) {
        Q = enow;
        enow = e[++eindex];
    } else {
        Q = fnow;
        fnow = f[++findex];
    }
    hindex = 0;
    if ((eindex < elen) && (findex < flen)) {
        if ((fnow > enow) == (fnow > -enow)) {
            Fast_Two_Sum(enow, Q, Qnew, hh);
            enow = e[++eindex];
        } else {
            Fast_Two_Sum(fnow, Q, Qnew, hh);
            fnow = f[++findex];
        }
        Q = Qnew;
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
        while ((eindex < elen) && (findex < flen)) {
            if ((fnow > enow) == (fnow > -enow)) {
                Two_Sum(Q, enow, Qnew, hh);
                enow = e[++eindex];
            } else {
                Two_Sum(Q, fnow, Qnew, hh);
                fnow = f[++findex];
            }
            Q = Qnew;
            if (hh != 0.0) {
                h[hindex++] = hh;
            }
        }
    }
    while (eindex < elen) {
        Two_Sum(Q, enow, Qnew, hh);
        enow = e[++eindex];
        Q = Qnew;
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
    }
    while (findex < flen) {
        Two_Sum(Q, fnow, Qnew, hh);
        fnow = f[++findex];
        Q = Qnew;
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
    }
    if ((Q != 0.0) || (hindex == 0)) {
        h[hindex++] = Q;
    }
    return hindex;
}

//*****************************************************************************
//*                                                                           *
//*  gpudt_scale_expansion_zeroelim()   Multiply an expansion by a scalar,          *
//*                               eliminating zero components from the        *
//*                               output expansion.                           *
//*                                                                           *
//*  Sets h = be.  See my Robust Predicates paper for details.                *
//*                                                                           *
//*  Maintains the nonoverlapping property.  If round-to-even is used (as     *
//*  with IEEE 754), maintains the strongly nonoverlapping and nonadjacent    *
//*  properties as well.  (That is, if e has one of these properties, so      *
//*  will h.)                                                                 *
//*                                                                           *
//*****************************************************************************

int gpudt_scale_expansion_zeroelim(int elen, REAL *e, REAL b, REAL *h)
{
    INEXACT REAL Q, sum;
    REAL hh;
    INEXACT REAL product1;
    REAL product0;
    int eindex, hindex;
    REAL enow;
    INEXACT REAL bvirt;
    REAL avirt, bround, around;
    INEXACT REAL c;
    INEXACT REAL abig;
    REAL ahi, alo, bhi, blo;
    REAL err1, err2, err3;

    Split(b, bhi, blo);
    Two_Product_Presplit(e[0], b, bhi, blo, Q, hh);
    hindex = 0;
    if (hh != 0) {
        h[hindex++] = hh;
    }
    for (eindex = 1; eindex < elen; eindex++) {
        enow = e[eindex];
        Two_Product_Presplit(enow, b, bhi, blo, product1, product0);
        Two_Sum(Q, product0, sum, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }
        Fast_Two_Sum(product1, sum, Q, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }
    }
    if ((Q != 0.0) || (hindex == 0)) {
        h[hindex++] = Q;
    }
    return hindex;
}

//*****************************************************************************
//*                                                                           *
//*  gpudt_estimate()   Produce a one-word gpudt_estimate of an expansion's value.        *
//*                                                                           *
//*  See my Robust Predicates paper for details.                              *
//*                                                                           *
//*****************************************************************************

REAL gpudt_estimate(int elen, REAL *e)
{
    REAL Q;
    int eindex;

    Q = e[0];
    for (eindex = 1; eindex < elen; eindex++) {
        Q += e[eindex];
    }
    return Q;
}

//*****************************************************************************
//*                                                                           *
//*  counterclockwise()   Return a positive value if the points pa, pb, and   *
//*                       pc occur in counterclockwise order; a negative      *
//*                       value if they occur in clockwise order; and zero    *
//*                       if they are collinear.  The result is also a rough  *
//*                       approximation of twice the signed area of the       *
//*                       triangle defined by the three points.               *
//*                                                                           *
//*  Uses exact arithmetic if necessary to ensure a correct answer.  The      *
//*  result returned is the determinant of a matrix.  This determinant is     *
//*  computed adaptively, in the sense that exact arithmetic is used only to  *
//*  the degree it is needed to ensure that the returned value has the        *
//*  correct sign.  Hence, this function is usually quite fast, but will run  *
//*  more slowly when the input points are collinear or nearly so.            *
//*                                                                           *
//*  See my Robust Predicates paper for details.                              *
//*                                                                           *
//*****************************************************************************
REAL counterclockwiseadapt(gpudtVertex *pa, gpudtVertex *pb, gpudtVertex *pc, REAL detsum)
{
    INEXACT REAL acx, acy, bcx, bcy;
    REAL acxtail, acytail, bcxtail, bcytail;
    INEXACT REAL detleft, detright;
    REAL detlefttail, detrighttail;
    REAL det, errbound;
    REAL B[4], C1[8], C2[12], D[16];
    INEXACT REAL B3;
    int C1length, C2length, Dlength;
    REAL u[4];
    INEXACT REAL u3;
    INEXACT REAL s1, t1;
    REAL s0, t0;

    INEXACT REAL bvirt;
    REAL avirt, bround, around;
    INEXACT REAL c;
    INEXACT REAL abig;
    REAL ahi, alo, bhi, blo;
    REAL err1, err2, err3;
    INEXACT REAL _i, _j;
    REAL _0;

    acx = (REAL) (pa->x - pc->x);
    bcx = (REAL) (pb->x - pc->x);
    acy = (REAL) (pa->y - pc->y);
    bcy = (REAL) (pb->y - pc->y);

    Two_Product(acx, bcy, detleft, detlefttail);
    Two_Product(acy, bcx, detright, detrighttail);

    Two_Two_Diff(detleft, detlefttail, detright, detrighttail,
        B3, B[2], B[1], B[0]);
    B[3] = B3;

    det = gpudt_estimate(4, B);
    errbound = gpudt_ccwerrboundB * detsum;
    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    Two_Diff_Tail(pa->x, pc->x, acx, acxtail);
    Two_Diff_Tail(pb->x, pc->x, bcx, bcxtail);
    Two_Diff_Tail(pa->y, pc->y, acy, acytail);
    Two_Diff_Tail(pb->y, pc->y, bcy, bcytail);

    if ((acxtail == 0.0) && (acytail == 0.0)
        && (bcxtail == 0.0) && (bcytail == 0.0)) {
            return det;
    }

    errbound = gpudt_ccwerrboundC * detsum + gpudt_resulterrbound * Absolute(det);
    det += (acx * bcytail + bcy * acxtail)
        - (acy * bcxtail + bcx * acytail);
    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    Two_Product(acxtail, bcy, s1, s0);
    Two_Product(acytail, bcx, t1, t0);
    Two_Two_Diff(s1, s0, t1, t0, u3, u[2], u[1], u[0]);
    u[3] = u3;
    C1length = gpudt_fast_expansion_sum_zeroelim(4, B, 4, u, C1);

    Two_Product(acx, bcytail, s1, s0);
    Two_Product(acy, bcxtail, t1, t0);
    Two_Two_Diff(s1, s0, t1, t0, u3, u[2], u[1], u[0]);
    u[3] = u3;
    C2length = gpudt_fast_expansion_sum_zeroelim(C1length, C1, 4, u, C2);

    Two_Product(acxtail, bcytail, s1, s0);
    Two_Product(acytail, bcxtail, t1, t0);
    Two_Two_Diff(s1, s0, t1, t0, u3, u[2], u[1], u[0]);
    u[3] = u3;
    Dlength = gpudt_fast_expansion_sum_zeroelim(C2length, C2, 4, u, D);

    return(D[Dlength - 1]);
}

REAL counterclockwise(gpudtVertex *pa, gpudtVertex *pb, gpudtVertex *pc)
{
    REAL detleft, detright, det;
    REAL detsum, errbound;

    detleft = (pa->x - pc->x) * (pb->y - pc->y);
    detright = (pa->y - pc->y) * (pb->x - pc->x);
    det = detleft - detright;

    if (detleft > 0.0) {
        if (detright <= 0.0) {
            return det;
        } else {
            detsum = detleft + detright;
        }
    } else if (detleft < 0.0) {
        if (detright >= 0.0) {
            return det;
        } else {
            detsum = -detleft - detright;
        }
    } else {
        return det;
    }

    errbound = gpudt_ccwerrboundA * detsum;
    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    return counterclockwiseadapt(pa, pb, pc, detsum);
}

//*****************************************************************************
//*                                                                           *
//*  incircle()   Return a positive value if the point pd lies inside the     *
//*               circle passing through pa, pb, and pc; a negative value if  *
//*               it lies outside; and zero if the four points are cocircular.*
//*               The points pa, pb, and pc must be in counterclockwise       *
//*               order, or the sign of the result will be reversed.          *
//*                                                                           *
//*  Uses exact arithmetic if necessary to ensure a correct answer.  The      *
//*  result returned is the determinant of a matrix.  This determinant is     *
//*  computed adaptively, in the sense that exact arithmetic is used only to  *
//*  the degree it is needed to ensure that the returned value has the        *
//*  correct sign.  Hence, this function is usually quite fast, but will run  *
//*  more slowly when the input points are cocircular or nearly so.           *
//*                                                                           *
//*  See my Robust Predicates paper for details.                              *
//*                                                                           *
//*****************************************************************************
REAL incircleadapt(gpudtVertex *pa, gpudtVertex *pb, gpudtVertex *pc, gpudtVertex *pd, REAL permanent)
{
    INEXACT REAL adx, bdx, cdx, ady, bdy, cdy;
    REAL det, errbound;

    INEXACT REAL bdxcdy1, cdxbdy1, cdxady1, adxcdy1, adxbdy1, bdxady1;
    REAL bdxcdy0, cdxbdy0, cdxady0, adxcdy0, adxbdy0, bdxady0;
    REAL bc[4], ca[4], ab[4];
    INEXACT REAL bc3, ca3, ab3;
    REAL axbc[8], axxbc[16], aybc[8], ayybc[16], adet[32];
    int axbclen, axxbclen, aybclen, ayybclen, alen;
    REAL bxca[8], bxxca[16], byca[8], byyca[16], bdet[32];
    int bxcalen, bxxcalen, bycalen, byycalen, blen;
    REAL cxab[8], cxxab[16], cyab[8], cyyab[16], cdet[32];
    int cxablen, cxxablen, cyablen, cyyablen, clen;
    REAL abdet[64];
    int ablen;
    REAL fin1[1152], fin2[1152];
    REAL *finnow, *finother, *finswap;
    int finlength;

    REAL adxtail, bdxtail, cdxtail, adytail, bdytail, cdytail;
    INEXACT REAL adxadx1, adyady1, bdxbdx1, bdybdy1, cdxcdx1, cdycdy1;
    REAL adxadx0, adyady0, bdxbdx0, bdybdy0, cdxcdx0, cdycdy0;
    REAL aa[4], bb[4], cc[4];
    INEXACT REAL aa3, bb3, cc3;
    INEXACT REAL ti1, tj1;
    REAL ti0, tj0;
    REAL u[4], v[4];
    INEXACT REAL u3, v3;
    REAL temp8[8], temp16a[16], temp16b[16], temp16c[16];
    REAL temp32a[32], temp32b[32], temp48[48], temp64[64];
    int temp8len, temp16alen, temp16blen, temp16clen;
    int temp32alen, temp32blen, temp48len, temp64len;
    REAL axtbb[8], axtcc[8], aytbb[8], aytcc[8];
    int axtbblen, axtcclen, aytbblen, aytcclen;
    REAL bxtaa[8], bxtcc[8], bytaa[8], bytcc[8];
    int bxtaalen, bxtcclen, bytaalen, bytcclen;
    REAL cxtaa[8], cxtbb[8], cytaa[8], cytbb[8];
    int cxtaalen, cxtbblen, cytaalen, cytbblen;
    REAL axtbc[8], aytbc[8], bxtca[8], bytca[8], cxtab[8], cytab[8];
    int axtbclen, aytbclen, bxtcalen, bytcalen, cxtablen, cytablen;
    REAL axtbct[16], aytbct[16], bxtcat[16], bytcat[16], cxtabt[16], cytabt[16];
    int axtbctlen, aytbctlen, bxtcatlen, bytcatlen, cxtabtlen, cytabtlen;
    REAL axtbctt[8], aytbctt[8], bxtcatt[8];
    REAL bytcatt[8], cxtabtt[8], cytabtt[8];
    int axtbcttlen, aytbcttlen, bxtcattlen, bytcattlen, cxtabttlen, cytabttlen;
    REAL abt[8], bct[8], cat[8];
    int abtlen, bctlen, catlen;
    REAL abtt[4], bctt[4], catt[4];
    int abttlen, bcttlen, cattlen;
    INEXACT REAL abtt3, bctt3, catt3;
    REAL negate;

    INEXACT REAL bvirt;
    REAL avirt, bround, around;
    INEXACT REAL c;
    INEXACT REAL abig;
    REAL ahi, alo, bhi, blo;
    REAL err1, err2, err3;
    INEXACT REAL _i, _j;
    REAL _0;

    adx = (REAL) (pa->x - pd->x);
    bdx = (REAL) (pb->x - pd->x);
    cdx = (REAL) (pc->x - pd->x);
    ady = (REAL) (pa->y - pd->y);
    bdy = (REAL) (pb->y - pd->y);
    cdy = (REAL) (pc->y - pd->y);

    Two_Product(bdx, cdy, bdxcdy1, bdxcdy0);
    Two_Product(cdx, bdy, cdxbdy1, cdxbdy0);
    Two_Two_Diff(bdxcdy1, bdxcdy0, cdxbdy1, cdxbdy0, bc3, bc[2], bc[1], bc[0]);
    bc[3] = bc3;
    axbclen = gpudt_scale_expansion_zeroelim(4, bc, adx, axbc);
    axxbclen = gpudt_scale_expansion_zeroelim(axbclen, axbc, adx, axxbc);
    aybclen = gpudt_scale_expansion_zeroelim(4, bc, ady, aybc);
    ayybclen = gpudt_scale_expansion_zeroelim(aybclen, aybc, ady, ayybc);
    alen = gpudt_fast_expansion_sum_zeroelim(axxbclen, axxbc, ayybclen, ayybc, adet);

    Two_Product(cdx, ady, cdxady1, cdxady0);
    Two_Product(adx, cdy, adxcdy1, adxcdy0);
    Two_Two_Diff(cdxady1, cdxady0, adxcdy1, adxcdy0, ca3, ca[2], ca[1], ca[0]);
    ca[3] = ca3;
    bxcalen = gpudt_scale_expansion_zeroelim(4, ca, bdx, bxca);
    bxxcalen = gpudt_scale_expansion_zeroelim(bxcalen, bxca, bdx, bxxca);
    bycalen = gpudt_scale_expansion_zeroelim(4, ca, bdy, byca);
    byycalen = gpudt_scale_expansion_zeroelim(bycalen, byca, bdy, byyca);
    blen = gpudt_fast_expansion_sum_zeroelim(bxxcalen, bxxca, byycalen, byyca, bdet);

    Two_Product(adx, bdy, adxbdy1, adxbdy0);
    Two_Product(bdx, ady, bdxady1, bdxady0);
    Two_Two_Diff(adxbdy1, adxbdy0, bdxady1, bdxady0, ab3, ab[2], ab[1], ab[0]);
    ab[3] = ab3;
    cxablen = gpudt_scale_expansion_zeroelim(4, ab, cdx, cxab);
    cxxablen = gpudt_scale_expansion_zeroelim(cxablen, cxab, cdx, cxxab);
    cyablen = gpudt_scale_expansion_zeroelim(4, ab, cdy, cyab);
    cyyablen = gpudt_scale_expansion_zeroelim(cyablen, cyab, cdy, cyyab);
    clen = gpudt_fast_expansion_sum_zeroelim(cxxablen, cxxab, cyyablen, cyyab, cdet);

    ablen = gpudt_fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);
    finlength = gpudt_fast_expansion_sum_zeroelim(ablen, abdet, clen, cdet, fin1);

    det = gpudt_estimate(finlength, fin1);
    errbound = gpudt_iccerrboundB * permanent;
    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    Two_Diff_Tail(pa->x, pd->x, adx, adxtail);
    Two_Diff_Tail(pa->y, pd->y, ady, adytail);
    Two_Diff_Tail(pb->x, pd->x, bdx, bdxtail);
    Two_Diff_Tail(pb->y, pd->y, bdy, bdytail);
    Two_Diff_Tail(pc->x, pd->x, cdx, cdxtail);
    Two_Diff_Tail(pc->y, pd->y, cdy, cdytail);
    if ((adxtail == 0.0) && (bdxtail == 0.0) && (cdxtail == 0.0)
        && (adytail == 0.0) && (bdytail == 0.0) && (cdytail == 0.0)) {
            return det;
    }

    errbound = gpudt_iccerrboundC * permanent + gpudt_resulterrbound * Absolute(det);
    det += ((adx * adx + ady * ady) * ((bdx * cdytail + cdy * bdxtail)
        - (bdy * cdxtail + cdx * bdytail))
        + 2.0 * (adx * adxtail + ady * adytail) * (bdx * cdy - bdy * cdx))
        + ((bdx * bdx + bdy * bdy) * ((cdx * adytail + ady * cdxtail)
        - (cdy * adxtail + adx * cdytail))
        + 2.0 * (bdx * bdxtail + bdy * bdytail) * (cdx * ady - cdy * adx))
        + ((cdx * cdx + cdy * cdy) * ((adx * bdytail + bdy * adxtail)
        - (ady * bdxtail + bdx * adytail))
        + 2.0 * (cdx * cdxtail + cdy * cdytail) * (adx * bdy - ady * bdx));
    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    finnow = fin1;
    finother = fin2;

    if ((bdxtail != 0.0) || (bdytail != 0.0)
        || (cdxtail != 0.0) || (cdytail != 0.0)) {
            Square(adx, adxadx1, adxadx0);
            Square(ady, adyady1, adyady0);
            Two_Two_Sum(adxadx1, adxadx0, adyady1, adyady0, aa3, aa[2], aa[1], aa[0]);
            aa[3] = aa3;
    }
    if ((cdxtail != 0.0) || (cdytail != 0.0)
        || (adxtail != 0.0) || (adytail != 0.0)) {
            Square(bdx, bdxbdx1, bdxbdx0);
            Square(bdy, bdybdy1, bdybdy0);
            Two_Two_Sum(bdxbdx1, bdxbdx0, bdybdy1, bdybdy0, bb3, bb[2], bb[1], bb[0]);
            bb[3] = bb3;
    }
    if ((adxtail != 0.0) || (adytail != 0.0)
        || (bdxtail != 0.0) || (bdytail != 0.0)) {
            Square(cdx, cdxcdx1, cdxcdx0);
            Square(cdy, cdycdy1, cdycdy0);
            Two_Two_Sum(cdxcdx1, cdxcdx0, cdycdy1, cdycdy0, cc3, cc[2], cc[1], cc[0]);
            cc[3] = cc3;
    }

    if (adxtail != 0.0) {
        axtbclen = gpudt_scale_expansion_zeroelim(4, bc, adxtail, axtbc);
        temp16alen = gpudt_scale_expansion_zeroelim(axtbclen, axtbc, 2.0 * adx,
            temp16a);

        axtcclen = gpudt_scale_expansion_zeroelim(4, cc, adxtail, axtcc);
        temp16blen = gpudt_scale_expansion_zeroelim(axtcclen, axtcc, bdy, temp16b);

        axtbblen = gpudt_scale_expansion_zeroelim(4, bb, adxtail, axtbb);
        temp16clen = gpudt_scale_expansion_zeroelim(axtbblen, axtbb, -cdy, temp16c);

        temp32alen = gpudt_fast_expansion_sum_zeroelim(temp16alen, temp16a,
            temp16blen, temp16b, temp32a);
        temp48len = gpudt_fast_expansion_sum_zeroelim(temp16clen, temp16c,
            temp32alen, temp32a, temp48);
        finlength = gpudt_fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
            temp48, finother);
        finswap = finnow; finnow = finother; finother = finswap;
    }
    if (adytail != 0.0) {
        aytbclen = gpudt_scale_expansion_zeroelim(4, bc, adytail, aytbc);
        temp16alen = gpudt_scale_expansion_zeroelim(aytbclen, aytbc, 2.0 * ady,
            temp16a);

        aytbblen = gpudt_scale_expansion_zeroelim(4, bb, adytail, aytbb);
        temp16blen = gpudt_scale_expansion_zeroelim(aytbblen, aytbb, cdx, temp16b);

        aytcclen = gpudt_scale_expansion_zeroelim(4, cc, adytail, aytcc);
        temp16clen = gpudt_scale_expansion_zeroelim(aytcclen, aytcc, -bdx, temp16c);

        temp32alen = gpudt_fast_expansion_sum_zeroelim(temp16alen, temp16a,
            temp16blen, temp16b, temp32a);
        temp48len = gpudt_fast_expansion_sum_zeroelim(temp16clen, temp16c,
            temp32alen, temp32a, temp48);
        finlength = gpudt_fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
            temp48, finother);
        finswap = finnow; finnow = finother; finother = finswap;
    }
    if (bdxtail != 0.0) {
        bxtcalen = gpudt_scale_expansion_zeroelim(4, ca, bdxtail, bxtca);
        temp16alen = gpudt_scale_expansion_zeroelim(bxtcalen, bxtca, 2.0 * bdx,
            temp16a);

        bxtaalen = gpudt_scale_expansion_zeroelim(4, aa, bdxtail, bxtaa);
        temp16blen = gpudt_scale_expansion_zeroelim(bxtaalen, bxtaa, cdy, temp16b);

        bxtcclen = gpudt_scale_expansion_zeroelim(4, cc, bdxtail, bxtcc);
        temp16clen = gpudt_scale_expansion_zeroelim(bxtcclen, bxtcc, -ady, temp16c);

        temp32alen = gpudt_fast_expansion_sum_zeroelim(temp16alen, temp16a,
            temp16blen, temp16b, temp32a);
        temp48len = gpudt_fast_expansion_sum_zeroelim(temp16clen, temp16c,
            temp32alen, temp32a, temp48);
        finlength = gpudt_fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
            temp48, finother);
        finswap = finnow; finnow = finother; finother = finswap;
    }
    if (bdytail != 0.0) {
        bytcalen = gpudt_scale_expansion_zeroelim(4, ca, bdytail, bytca);
        temp16alen = gpudt_scale_expansion_zeroelim(bytcalen, bytca, 2.0 * bdy,
            temp16a);

        bytcclen = gpudt_scale_expansion_zeroelim(4, cc, bdytail, bytcc);
        temp16blen = gpudt_scale_expansion_zeroelim(bytcclen, bytcc, adx, temp16b);

        bytaalen = gpudt_scale_expansion_zeroelim(4, aa, bdytail, bytaa);
        temp16clen = gpudt_scale_expansion_zeroelim(bytaalen, bytaa, -cdx, temp16c);

        temp32alen = gpudt_fast_expansion_sum_zeroelim(temp16alen, temp16a,
            temp16blen, temp16b, temp32a);
        temp48len = gpudt_fast_expansion_sum_zeroelim(temp16clen, temp16c,
            temp32alen, temp32a, temp48);
        finlength = gpudt_fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
            temp48, finother);
        finswap = finnow; finnow = finother; finother = finswap;
    }
    if (cdxtail != 0.0) {
        cxtablen = gpudt_scale_expansion_zeroelim(4, ab, cdxtail, cxtab);
        temp16alen = gpudt_scale_expansion_zeroelim(cxtablen, cxtab, 2.0 * cdx,
            temp16a);

        cxtbblen = gpudt_scale_expansion_zeroelim(4, bb, cdxtail, cxtbb);
        temp16blen = gpudt_scale_expansion_zeroelim(cxtbblen, cxtbb, ady, temp16b);

        cxtaalen = gpudt_scale_expansion_zeroelim(4, aa, cdxtail, cxtaa);
        temp16clen = gpudt_scale_expansion_zeroelim(cxtaalen, cxtaa, -bdy, temp16c);

        temp32alen = gpudt_fast_expansion_sum_zeroelim(temp16alen, temp16a,
            temp16blen, temp16b, temp32a);
        temp48len = gpudt_fast_expansion_sum_zeroelim(temp16clen, temp16c,
            temp32alen, temp32a, temp48);
        finlength = gpudt_fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
            temp48, finother);
        finswap = finnow; finnow = finother; finother = finswap;
    }
    if (cdytail != 0.0) {
        cytablen = gpudt_scale_expansion_zeroelim(4, ab, cdytail, cytab);
        temp16alen = gpudt_scale_expansion_zeroelim(cytablen, cytab, 2.0 * cdy,
            temp16a);

        cytaalen = gpudt_scale_expansion_zeroelim(4, aa, cdytail, cytaa);
        temp16blen = gpudt_scale_expansion_zeroelim(cytaalen, cytaa, bdx, temp16b);

        cytbblen = gpudt_scale_expansion_zeroelim(4, bb, cdytail, cytbb);
        temp16clen = gpudt_scale_expansion_zeroelim(cytbblen, cytbb, -adx, temp16c);

        temp32alen = gpudt_fast_expansion_sum_zeroelim(temp16alen, temp16a,
            temp16blen, temp16b, temp32a);
        temp48len = gpudt_fast_expansion_sum_zeroelim(temp16clen, temp16c,
            temp32alen, temp32a, temp48);
        finlength = gpudt_fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
            temp48, finother);
        finswap = finnow; finnow = finother; finother = finswap;
    }

    if ((adxtail != 0.0) || (adytail != 0.0)) {
        if ((bdxtail != 0.0) || (bdytail != 0.0)
            || (cdxtail != 0.0) || (cdytail != 0.0)) {
                Two_Product(bdxtail, cdy, ti1, ti0);
                Two_Product(bdx, cdytail, tj1, tj0);
                Two_Two_Sum(ti1, ti0, tj1, tj0, u3, u[2], u[1], u[0]);
                u[3] = u3;
                negate = -bdy;
                Two_Product(cdxtail, negate, ti1, ti0);
                negate = -bdytail;
                Two_Product(cdx, negate, tj1, tj0);
                Two_Two_Sum(ti1, ti0, tj1, tj0, v3, v[2], v[1], v[0]);
                v[3] = v3;
                bctlen = gpudt_fast_expansion_sum_zeroelim(4, u, 4, v, bct);

                Two_Product(bdxtail, cdytail, ti1, ti0);
                Two_Product(cdxtail, bdytail, tj1, tj0);
                Two_Two_Diff(ti1, ti0, tj1, tj0, bctt3, bctt[2], bctt[1], bctt[0]);
                bctt[3] = bctt3;
                bcttlen = 4;
        } else {
            bct[0] = 0.0;
            bctlen = 1;
            bctt[0] = 0.0;
            bcttlen = 1;
        }

        if (adxtail != 0.0) {
            temp16alen = gpudt_scale_expansion_zeroelim(axtbclen, axtbc, adxtail, temp16a);
            axtbctlen = gpudt_scale_expansion_zeroelim(bctlen, bct, adxtail, axtbct);
            temp32alen = gpudt_scale_expansion_zeroelim(axtbctlen, axtbct, 2.0 * adx,
                temp32a);
            temp48len = gpudt_fast_expansion_sum_zeroelim(temp16alen, temp16a,
                temp32alen, temp32a, temp48);
            finlength = gpudt_fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                temp48, finother);
            finswap = finnow; finnow = finother; finother = finswap;
            if (bdytail != 0.0) {
                temp8len = gpudt_scale_expansion_zeroelim(4, cc, adxtail, temp8);
                temp16alen = gpudt_scale_expansion_zeroelim(temp8len, temp8, bdytail,
                    temp16a);
                finlength = gpudt_fast_expansion_sum_zeroelim(finlength, finnow, temp16alen,
                    temp16a, finother);
                finswap = finnow; finnow = finother; finother = finswap;
            }
            if (cdytail != 0.0) {
                temp8len = gpudt_scale_expansion_zeroelim(4, bb, -adxtail, temp8);
                temp16alen = gpudt_scale_expansion_zeroelim(temp8len, temp8, cdytail,
                    temp16a);
                finlength = gpudt_fast_expansion_sum_zeroelim(finlength, finnow, temp16alen,
                    temp16a, finother);
                finswap = finnow; finnow = finother; finother = finswap;
            }

            temp32alen = gpudt_scale_expansion_zeroelim(axtbctlen, axtbct, adxtail,
                temp32a);
            axtbcttlen = gpudt_scale_expansion_zeroelim(bcttlen, bctt, adxtail, axtbctt);
            temp16alen = gpudt_scale_expansion_zeroelim(axtbcttlen, axtbctt, 2.0 * adx,
                temp16a);
            temp16blen = gpudt_scale_expansion_zeroelim(axtbcttlen, axtbctt, adxtail,
                temp16b);
            temp32blen = gpudt_fast_expansion_sum_zeroelim(temp16alen, temp16a,
                temp16blen, temp16b, temp32b);
            temp64len = gpudt_fast_expansion_sum_zeroelim(temp32alen, temp32a,
                temp32blen, temp32b, temp64);
            finlength = gpudt_fast_expansion_sum_zeroelim(finlength, finnow, temp64len,
                temp64, finother);
            finswap = finnow; finnow = finother; finother = finswap;
        }
        if (adytail != 0.0) {
            temp16alen = gpudt_scale_expansion_zeroelim(aytbclen, aytbc, adytail, temp16a);
            aytbctlen = gpudt_scale_expansion_zeroelim(bctlen, bct, adytail, aytbct);
            temp32alen = gpudt_scale_expansion_zeroelim(aytbctlen, aytbct, 2.0 * ady,
                temp32a);
            temp48len = gpudt_fast_expansion_sum_zeroelim(temp16alen, temp16a,
                temp32alen, temp32a, temp48);
            finlength = gpudt_fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                temp48, finother);
            finswap = finnow; finnow = finother; finother = finswap;


            temp32alen = gpudt_scale_expansion_zeroelim(aytbctlen, aytbct, adytail,
                temp32a);
            aytbcttlen = gpudt_scale_expansion_zeroelim(bcttlen, bctt, adytail, aytbctt);
            temp16alen = gpudt_scale_expansion_zeroelim(aytbcttlen, aytbctt, 2.0 * ady,
                temp16a);
            temp16blen = gpudt_scale_expansion_zeroelim(aytbcttlen, aytbctt, adytail,
                temp16b);
            temp32blen = gpudt_fast_expansion_sum_zeroelim(temp16alen, temp16a,
                temp16blen, temp16b, temp32b);
            temp64len = gpudt_fast_expansion_sum_zeroelim(temp32alen, temp32a,
                temp32blen, temp32b, temp64);
            finlength = gpudt_fast_expansion_sum_zeroelim(finlength, finnow, temp64len,
                temp64, finother);
            finswap = finnow; finnow = finother; finother = finswap;
        }
    }
    if ((bdxtail != 0.0) || (bdytail != 0.0)) {
        if ((cdxtail != 0.0) || (cdytail != 0.0)
            || (adxtail != 0.0) || (adytail != 0.0)) {
                Two_Product(cdxtail, ady, ti1, ti0);
                Two_Product(cdx, adytail, tj1, tj0);
                Two_Two_Sum(ti1, ti0, tj1, tj0, u3, u[2], u[1], u[0]);
                u[3] = u3;
                negate = -cdy;
                Two_Product(adxtail, negate, ti1, ti0);
                negate = -cdytail;
                Two_Product(adx, negate, tj1, tj0);
                Two_Two_Sum(ti1, ti0, tj1, tj0, v3, v[2], v[1], v[0]);
                v[3] = v3;
                catlen = gpudt_fast_expansion_sum_zeroelim(4, u, 4, v, cat);

                Two_Product(cdxtail, adytail, ti1, ti0);
                Two_Product(adxtail, cdytail, tj1, tj0);
                Two_Two_Diff(ti1, ti0, tj1, tj0, catt3, catt[2], catt[1], catt[0]);
                catt[3] = catt3;
                cattlen = 4;
        } else {
            cat[0] = 0.0;
            catlen = 1;
            catt[0] = 0.0;
            cattlen = 1;
        }

        if (bdxtail != 0.0) {
            temp16alen = gpudt_scale_expansion_zeroelim(bxtcalen, bxtca, bdxtail, temp16a);
            bxtcatlen = gpudt_scale_expansion_zeroelim(catlen, cat, bdxtail, bxtcat);
            temp32alen = gpudt_scale_expansion_zeroelim(bxtcatlen, bxtcat, 2.0 * bdx,
                temp32a);
            temp48len = gpudt_fast_expansion_sum_zeroelim(temp16alen, temp16a,
                temp32alen, temp32a, temp48);
            finlength = gpudt_fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                temp48, finother);
            finswap = finnow; finnow = finother; finother = finswap;
            if (cdytail != 0.0) {
                temp8len = gpudt_scale_expansion_zeroelim(4, aa, bdxtail, temp8);
                temp16alen = gpudt_scale_expansion_zeroelim(temp8len, temp8, cdytail,
                    temp16a);
                finlength = gpudt_fast_expansion_sum_zeroelim(finlength, finnow, temp16alen,
                    temp16a, finother);
                finswap = finnow; finnow = finother; finother = finswap;
            }
            if (adytail != 0.0) {
                temp8len = gpudt_scale_expansion_zeroelim(4, cc, -bdxtail, temp8);
                temp16alen = gpudt_scale_expansion_zeroelim(temp8len, temp8, adytail,
                    temp16a);
                finlength = gpudt_fast_expansion_sum_zeroelim(finlength, finnow, temp16alen,
                    temp16a, finother);
                finswap = finnow; finnow = finother; finother = finswap;
            }

            temp32alen = gpudt_scale_expansion_zeroelim(bxtcatlen, bxtcat, bdxtail,
                temp32a);
            bxtcattlen = gpudt_scale_expansion_zeroelim(cattlen, catt, bdxtail, bxtcatt);
            temp16alen = gpudt_scale_expansion_zeroelim(bxtcattlen, bxtcatt, 2.0 * bdx,
                temp16a);
            temp16blen = gpudt_scale_expansion_zeroelim(bxtcattlen, bxtcatt, bdxtail,
                temp16b);
            temp32blen = gpudt_fast_expansion_sum_zeroelim(temp16alen, temp16a,
                temp16blen, temp16b, temp32b);
            temp64len = gpudt_fast_expansion_sum_zeroelim(temp32alen, temp32a,
                temp32blen, temp32b, temp64);
            finlength = gpudt_fast_expansion_sum_zeroelim(finlength, finnow, temp64len,
                temp64, finother);
            finswap = finnow; finnow = finother; finother = finswap;
        }
        if (bdytail != 0.0) {
            temp16alen = gpudt_scale_expansion_zeroelim(bytcalen, bytca, bdytail, temp16a);
            bytcatlen = gpudt_scale_expansion_zeroelim(catlen, cat, bdytail, bytcat);
            temp32alen = gpudt_scale_expansion_zeroelim(bytcatlen, bytcat, 2.0 * bdy,
                temp32a);
            temp48len = gpudt_fast_expansion_sum_zeroelim(temp16alen, temp16a,
                temp32alen, temp32a, temp48);
            finlength = gpudt_fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                temp48, finother);
            finswap = finnow; finnow = finother; finother = finswap;


            temp32alen = gpudt_scale_expansion_zeroelim(bytcatlen, bytcat, bdytail,
                temp32a);
            bytcattlen = gpudt_scale_expansion_zeroelim(cattlen, catt, bdytail, bytcatt);
            temp16alen = gpudt_scale_expansion_zeroelim(bytcattlen, bytcatt, 2.0 * bdy,
                temp16a);
            temp16blen = gpudt_scale_expansion_zeroelim(bytcattlen, bytcatt, bdytail,
                temp16b);
            temp32blen = gpudt_fast_expansion_sum_zeroelim(temp16alen, temp16a,
                temp16blen, temp16b, temp32b);
            temp64len = gpudt_fast_expansion_sum_zeroelim(temp32alen, temp32a,
                temp32blen, temp32b, temp64);
            finlength = gpudt_fast_expansion_sum_zeroelim(finlength, finnow, temp64len,
                temp64, finother);
            finswap = finnow; finnow = finother; finother = finswap;
        }
    }
    if ((cdxtail != 0.0) || (cdytail != 0.0)) {
        if ((adxtail != 0.0) || (adytail != 0.0)
            || (bdxtail != 0.0) || (bdytail != 0.0)) {
                Two_Product(adxtail, bdy, ti1, ti0);
                Two_Product(adx, bdytail, tj1, tj0);
                Two_Two_Sum(ti1, ti0, tj1, tj0, u3, u[2], u[1], u[0]);
                u[3] = u3;
                negate = -ady;
                Two_Product(bdxtail, negate, ti1, ti0);
                negate = -adytail;
                Two_Product(bdx, negate, tj1, tj0);
                Two_Two_Sum(ti1, ti0, tj1, tj0, v3, v[2], v[1], v[0]);
                v[3] = v3;
                abtlen = gpudt_fast_expansion_sum_zeroelim(4, u, 4, v, abt);

                Two_Product(adxtail, bdytail, ti1, ti0);
                Two_Product(bdxtail, adytail, tj1, tj0);
                Two_Two_Diff(ti1, ti0, tj1, tj0, abtt3, abtt[2], abtt[1], abtt[0]);
                abtt[3] = abtt3;
                abttlen = 4;
        } else {
            abt[0] = 0.0;
            abtlen = 1;
            abtt[0] = 0.0;
            abttlen = 1;
        }

        if (cdxtail != 0.0) {
            temp16alen = gpudt_scale_expansion_zeroelim(cxtablen, cxtab, cdxtail, temp16a);
            cxtabtlen = gpudt_scale_expansion_zeroelim(abtlen, abt, cdxtail, cxtabt);
            temp32alen = gpudt_scale_expansion_zeroelim(cxtabtlen, cxtabt, 2.0 * cdx,
                temp32a);
            temp48len = gpudt_fast_expansion_sum_zeroelim(temp16alen, temp16a,
                temp32alen, temp32a, temp48);
            finlength = gpudt_fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                temp48, finother);
            finswap = finnow; finnow = finother; finother = finswap;
            if (adytail != 0.0) {
                temp8len = gpudt_scale_expansion_zeroelim(4, bb, cdxtail, temp8);
                temp16alen = gpudt_scale_expansion_zeroelim(temp8len, temp8, adytail,
                    temp16a);
                finlength = gpudt_fast_expansion_sum_zeroelim(finlength, finnow, temp16alen,
                    temp16a, finother);
                finswap = finnow; finnow = finother; finother = finswap;
            }
            if (bdytail != 0.0) {
                temp8len = gpudt_scale_expansion_zeroelim(4, aa, -cdxtail, temp8);
                temp16alen = gpudt_scale_expansion_zeroelim(temp8len, temp8, bdytail,
                    temp16a);
                finlength = gpudt_fast_expansion_sum_zeroelim(finlength, finnow, temp16alen,
                    temp16a, finother);
                finswap = finnow; finnow = finother; finother = finswap;
            }

            temp32alen = gpudt_scale_expansion_zeroelim(cxtabtlen, cxtabt, cdxtail,
                temp32a);
            cxtabttlen = gpudt_scale_expansion_zeroelim(abttlen, abtt, cdxtail, cxtabtt);
            temp16alen = gpudt_scale_expansion_zeroelim(cxtabttlen, cxtabtt, 2.0 * cdx,
                temp16a);
            temp16blen = gpudt_scale_expansion_zeroelim(cxtabttlen, cxtabtt, cdxtail,
                temp16b);
            temp32blen = gpudt_fast_expansion_sum_zeroelim(temp16alen, temp16a,
                temp16blen, temp16b, temp32b);
            temp64len = gpudt_fast_expansion_sum_zeroelim(temp32alen, temp32a,
                temp32blen, temp32b, temp64);
            finlength = gpudt_fast_expansion_sum_zeroelim(finlength, finnow, temp64len,
                temp64, finother);
            finswap = finnow; finnow = finother; finother = finswap;
        }
        if (cdytail != 0.0) {
            temp16alen = gpudt_scale_expansion_zeroelim(cytablen, cytab, cdytail, temp16a);
            cytabtlen = gpudt_scale_expansion_zeroelim(abtlen, abt, cdytail, cytabt);
            temp32alen = gpudt_scale_expansion_zeroelim(cytabtlen, cytabt, 2.0 * cdy,
                temp32a);
            temp48len = gpudt_fast_expansion_sum_zeroelim(temp16alen, temp16a,
                temp32alen, temp32a, temp48);
            finlength = gpudt_fast_expansion_sum_zeroelim(finlength, finnow, temp48len,
                temp48, finother);
            finswap = finnow; finnow = finother; finother = finswap;


            temp32alen = gpudt_scale_expansion_zeroelim(cytabtlen, cytabt, cdytail,
                temp32a);
            cytabttlen = gpudt_scale_expansion_zeroelim(abttlen, abtt, cdytail, cytabtt);
            temp16alen = gpudt_scale_expansion_zeroelim(cytabttlen, cytabtt, 2.0 * cdy,
                temp16a);
            temp16blen = gpudt_scale_expansion_zeroelim(cytabttlen, cytabtt, cdytail,
                temp16b);
            temp32blen = gpudt_fast_expansion_sum_zeroelim(temp16alen, temp16a,
                temp16blen, temp16b, temp32b);
            temp64len = gpudt_fast_expansion_sum_zeroelim(temp32alen, temp32a,
                temp32blen, temp32b, temp64);
            finlength = gpudt_fast_expansion_sum_zeroelim(finlength, finnow, temp64len,
                temp64, finother);
            finswap = finnow; finnow = finother; finother = finswap;
        }
    }

    return finnow[finlength - 1];
}

REAL incircle(gpudtVertex *pa, gpudtVertex *pb, gpudtVertex *pc, gpudtVertex *pd)
{
    REAL adx, bdx, cdx, ady, bdy, cdy;
    REAL bdxcdy, cdxbdy, cdxady, adxcdy, adxbdy, bdxady;
    REAL alift, blift, clift;
    REAL det;
    REAL permanent, errbound;

    adx = pa->x - pd->x;
    bdx = pb->x - pd->x;
    cdx = pc->x - pd->x;
    ady = pa->y - pd->y;
    bdy = pb->y - pd->y;
    cdy = pc->y - pd->y;

    bdxcdy = bdx * cdy;
    cdxbdy = cdx * bdy;
    alift = adx * adx + ady * ady;

    cdxady = cdx * ady;
    adxcdy = adx * cdy;
    blift = bdx * bdx + bdy * bdy;

    adxbdy = adx * bdy;
    bdxady = bdx * ady;
    clift = cdx * cdx + cdy * cdy;

    det = alift * (bdxcdy - cdxbdy)
        + blift * (cdxady - adxcdy)
        + clift * (adxbdy - bdxady);

    permanent = (Absolute(bdxcdy) + Absolute(cdxbdy)) * alift
        + (Absolute(cdxady) + Absolute(adxcdy)) * blift
        + (Absolute(adxbdy) + Absolute(bdxady)) * clift;
    errbound = gpudt_iccerrboundA * permanent;
    if ((det > errbound) || (-det > errbound)) {
        return det;
    }

    return incircleadapt(pa, pb, pc, pd, permanent);
}

