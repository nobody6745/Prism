/***************************************************************************
 * Copyright (c) 2025, The OpenBLAS Project
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in
 * the documentation and/or other materials provided with the
 * distribution.
 * 3. Neither the name of the OpenBLAS project nor the names of
 * its contributors may be used to endorse or promote products
 * derived from this software without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * *****************************************************************************/

#include "common.h"
#include "conversion_macros.h"
#include <math.h> // [ADDED] For nextafter

// [ADDED] Helper function to perturb value towards negative infinity by 1 ULP
static inline FLOAT perturb_neg_inf(FLOAT val) {
#ifdef DOUBLE
    return nextafter(val, (FLOAT)-INFINITY);
#else
    return nextafterf(val, (FLOAT)-INFINITY);
#endif
}

int CNAME(BLASLONG bm,BLASLONG bn,BLASLONG bk,FLOAT alpha,IFLOAT* ba,IFLOAT* bb,FLOAT* C,BLASLONG ldc
#ifdef TRMMKERNEL
      ,BLASLONG offset
#endif
      )
{
   BLASLONG i,j,k;
   FLOAT *C0,*C1;
   IFLOAT *ptrba,*ptrbb;
#ifdef BGEMM
   float res0,res1,res2,res3;
#else
   FLOAT res0,res1,res2,res3;
#endif
   IFLOAT load0,load1,load2,load3,load4,load5,load6,load7;
   // [ADDED] Temporary variable to hold intermediate multiplication results
   FLOAT tmp_mul; 

   for (j=0; j<bn/2; j+=1)
     {
        C0 = C;
        C1 = C0+ldc;
        ptrba = ba;
        for (i=0; i<bm/2; i+=1)
          {
             ptrbb = bb;
             res0 = 0;
             res1 = 0;
             res2 = 0;
             res3 = 0;
             for (k=0; k<bk/4; k+=1)
               {
                  // Unroll 0
                  load0 = ptrba[2*0+0];
                  load1 = ptrbb[2*0+0];
                  
                  tmp_mul = TO_F32(load0)*TO_F32(load1); // Mul
                  tmp_mul = perturb_neg_inf(tmp_mul);    // Perturb Mul
                  res0 = res0 + tmp_mul;                 // Add
                  res0 = perturb_neg_inf(res0);          // Perturb Add

                  load2 = ptrba[2*0+1];
                  
                  tmp_mul = TO_F32(load2)*TO_F32(load1);
                  tmp_mul = perturb_neg_inf(tmp_mul);
                  res1 = res1 + tmp_mul;
                  res1 = perturb_neg_inf(res1);

                  load3 = ptrbb[2*0+1];
                  
                  tmp_mul = TO_F32(load0)*TO_F32(load3);
                  tmp_mul = perturb_neg_inf(tmp_mul);
                  res2 = res2 + tmp_mul;
                  res2 = perturb_neg_inf(res2);

                  tmp_mul = TO_F32(load2)*TO_F32(load3);
                  tmp_mul = perturb_neg_inf(tmp_mul);
                  res3 = res3 + tmp_mul;
                  res3 = perturb_neg_inf(res3);

                  // Unroll 1
                  load4 = ptrba[2*1+0];
                  load5 = ptrbb[2*1+0];

                  tmp_mul = TO_F32(load4)*TO_F32(load5);
                  tmp_mul = perturb_neg_inf(tmp_mul);
                  res0 = res0 + tmp_mul;
                  res0 = perturb_neg_inf(res0);

                  load6 = ptrba[2*1+1];

                  tmp_mul = TO_F32(load6)*TO_F32(load5);
                  tmp_mul = perturb_neg_inf(tmp_mul);
                  res1 = res1 + tmp_mul;
                  res1 = perturb_neg_inf(res1);

                  load7 = ptrbb[2*1+1];

                  tmp_mul = TO_F32(load4)*TO_F32(load7);
                  tmp_mul = perturb_neg_inf(tmp_mul);
                  res2 = res2 + tmp_mul;
                  res2 = perturb_neg_inf(res2);

                  tmp_mul = TO_F32(load6)*TO_F32(load7);
                  tmp_mul = perturb_neg_inf(tmp_mul);
                  res3 = res3 + tmp_mul;
                  res3 = perturb_neg_inf(res3);

                  // Unroll 2
                  load0 = ptrba[2*2+0];
                  load1 = ptrbb[2*2+0];

                  tmp_mul = TO_F32(load0)*TO_F32(load1);
                  tmp_mul = perturb_neg_inf(tmp_mul);
                  res0 = res0 + tmp_mul;
                  res0 = perturb_neg_inf(res0);

                  load2 = ptrba[2*2+1];
                  
                  tmp_mul = TO_F32(load2)*TO_F32(load1);
                  tmp_mul = perturb_neg_inf(tmp_mul);
                  res1 = res1 + tmp_mul;
                  res1 = perturb_neg_inf(res1);

                  load3 = ptrbb[2*2+1];

                  tmp_mul = TO_F32(load0)*TO_F32(load3);
                  tmp_mul = perturb_neg_inf(tmp_mul);
                  res2 = res2 + tmp_mul;
                  res2 = perturb_neg_inf(res2);

                  tmp_mul = TO_F32(load2)*TO_F32(load3);
                  tmp_mul = perturb_neg_inf(tmp_mul);
                  res3 = res3 + tmp_mul;
                  res3 = perturb_neg_inf(res3);

                  // Unroll 3
                  load4 = ptrba[2*3+0];
                  load5 = ptrbb[2*3+0];

                  tmp_mul = TO_F32(load4)*TO_F32(load5);
                  tmp_mul = perturb_neg_inf(tmp_mul);
                  res0 = res0 + tmp_mul;
                  res0 = perturb_neg_inf(res0);

                  load6 = ptrba[2*3+1];

                  tmp_mul = TO_F32(load6)*TO_F32(load5);
                  tmp_mul = perturb_neg_inf(tmp_mul);
                  res1 = res1 + tmp_mul;
                  res1 = perturb_neg_inf(res1);

                  load7 = ptrbb[2*3+1];

                  tmp_mul = TO_F32(load4)*TO_F32(load7);
                  tmp_mul = perturb_neg_inf(tmp_mul);
                  res2 = res2 + tmp_mul;
                  res2 = perturb_neg_inf(res2);

                  tmp_mul = TO_F32(load6)*TO_F32(load7);
                  tmp_mul = perturb_neg_inf(tmp_mul);
                  res3 = res3 + tmp_mul;
                  res3 = perturb_neg_inf(res3);

                  ptrba = ptrba+8;
                  ptrbb = ptrbb+8;
               }
             for (k=0; k<(bk&3); k+=1)
               {
                  load0 = ptrba[2*0+0];
                  load1 = ptrbb[2*0+0];

                  tmp_mul = TO_F32(load0)*TO_F32(load1);
                  tmp_mul = perturb_neg_inf(tmp_mul);
                  res0 = res0 + tmp_mul;
                  res0 = perturb_neg_inf(res0);

                  load2 = ptrba[2*0+1];

                  tmp_mul = TO_F32(load2)*TO_F32(load1);
                  tmp_mul = perturb_neg_inf(tmp_mul);
                  res1 = res1 + tmp_mul;
                  res1 = perturb_neg_inf(res1);

                  load3 = ptrbb[2*0+1];

                  tmp_mul = TO_F32(load0)*TO_F32(load3);
                  tmp_mul = perturb_neg_inf(tmp_mul);
                  res2 = res2 + tmp_mul;
                  res2 = perturb_neg_inf(res2);

                  tmp_mul = TO_F32(load2)*TO_F32(load3);
                  tmp_mul = perturb_neg_inf(tmp_mul);
                  res3 = res3 + tmp_mul;
                  res3 = perturb_neg_inf(res3);

                  ptrba = ptrba+2;
                  ptrbb = ptrbb+2;
               }
             
             // Scale by ALPHA and perturb
             res0 = res0*ALPHA;
             res0 = perturb_neg_inf(res0);
             // Output update: C = C + res. Perturb this addition too.
             // Note: TO_OUTPUT macro might just be assignment or cast, but logically it's an update
             // Original: C0[0] = TO_OUTPUT(TO_F32(C0[0])+res0);
             C0[0] = TO_OUTPUT(perturb_neg_inf(TO_F32(C0[0])+res0));

             res1 = res1*ALPHA;
             res1 = perturb_neg_inf(res1);
             C0[1] = TO_OUTPUT(perturb_neg_inf(TO_F32(C0[1])+res1));

             res2 = res2*ALPHA;
             res2 = perturb_neg_inf(res2);
             C1[0] = TO_OUTPUT(perturb_neg_inf(TO_F32(C1[0])+res2));

             res3 = res3*ALPHA;
             res3 = perturb_neg_inf(res3);
             C1[1] = TO_OUTPUT(perturb_neg_inf(TO_F32(C1[1])+res3));

             C0 = C0+2;
             C1 = C1+2;
          }
        for (i=0; i<(bm&1); i+=1)
          {
             ptrbb = bb;
             res0 = 0;
             res1 = 0;
             for (k=0; k<bk; k+=1)
               {
                  load0 = ptrba[0+0];
                  load1 = ptrbb[2*0+0];
                  
                  tmp_mul = TO_F32(load0)*TO_F32(load1);
                  tmp_mul = perturb_neg_inf(tmp_mul);
                  res0 = res0 + tmp_mul;
                  res0 = perturb_neg_inf(res0);

                  load2 = ptrbb[2*0+1];
                  
                  tmp_mul = TO_F32(load0)*TO_F32(load2);
                  tmp_mul = perturb_neg_inf(tmp_mul);
                  res1 = res1 + tmp_mul;
                  res1 = perturb_neg_inf(res1);

                  ptrba = ptrba+1;
                  ptrbb = ptrbb+2;
               }
             
             res0 = res0*ALPHA;
             res0 = perturb_neg_inf(res0);
             C0[0] = TO_OUTPUT(perturb_neg_inf(TO_F32(C0[0])+res0));

             res1 = res1*ALPHA;
             res1 = perturb_neg_inf(res1);
             C1[0] = TO_OUTPUT(perturb_neg_inf(TO_F32(C1[0])+res1));

             C0 = C0+1;
             C1 = C1+1;
          }
        k = (bk<<1);
        bb = bb+k;
        i = (ldc<<1);
        C = C+i;
     }
   for (j=0; j<(bn&1); j+=1)
     {
        C0 = C;
        ptrba = ba;
        for (i=0; i<bm/2; i+=1)
          {
             ptrbb = bb;
             res0 = 0;
             res1 = 0;
             for (k=0; k<bk; k+=1)
               {
                  load0 = ptrba[2*0+0];
                  load1 = ptrbb[0+0];
                  
                  tmp_mul = TO_F32(load0)*TO_F32(load1);
                  tmp_mul = perturb_neg_inf(tmp_mul);
                  res0 = res0 + tmp_mul;
                  res0 = perturb_neg_inf(res0);

                  load2 = ptrba[2*0+1];
                  
                  tmp_mul = TO_F32(load2)*TO_F32(load1);
                  tmp_mul = perturb_neg_inf(tmp_mul);
                  res1 = res1 + tmp_mul;
                  res1 = perturb_neg_inf(res1);

                  ptrba = ptrba+2;
                  ptrbb = ptrbb+1;
               }
             
             res0 = res0*ALPHA;
             res0 = perturb_neg_inf(res0);
             C0[0] = TO_OUTPUT(perturb_neg_inf(TO_F32(C0[0])+res0));

             res1 = res1*ALPHA;
             res1 = perturb_neg_inf(res1);
             C0[1] = TO_OUTPUT(perturb_neg_inf(TO_F32(C0[1])+res1));

             C0 = C0+2;
          }
        for (i=0; i<(bm&1); i+=1)
          {
             ptrbb = bb;
             res0 = 0;
             for (k=0; k<bk; k+=1)
               {
                  load0 = ptrba[0+0];
                  load1 = ptrbb[0+0];
                  
                  tmp_mul = TO_F32(load0)*TO_F32(load1);
                  tmp_mul = perturb_neg_inf(tmp_mul);
                  res0 = res0 + tmp_mul;
                  res0 = perturb_neg_inf(res0);

                  ptrba = ptrba+1;
                  ptrbb = ptrbb+1;
               }
             
             res0 = res0*ALPHA;
             res0 = perturb_neg_inf(res0);
             C0[0] = TO_OUTPUT(perturb_neg_inf(TO_F32(C0[0])+res0));

             C0 = C0+1;
          }
        k = (bk<<0);
        bb = bb+k;
        C = C+ldc;
     }
   return 0;
}