*  subroutines used for URAT1 release utility programs
*
*  open_zfile    : open direct access, unformatted zone file
*  getistar      : get all data items of individual star (single record)
*  nx_byte_flip  : apply byte flip on index array
*  flip2         : flip 2 byte integer
*  flip4         : flip 4 byte integer
*  valid_range   : restrict R*8 data item to given min,max 
*  get_zone_range: declination range --> required zone numbers
*  get_ra_range  : RA range --> required index for RA bins (900 arcsec)
*  as2hms        : convert arcsec (RA,Dec) into hms format
*  sorti         : sort 2-dim integer array
*
*  141120 NZ taken from UCAC4 release utility code, adopt for URAT
*
************************************************************************

      SUBROUTINE open_zfile (pathz,un,zn,fnzone)
C
C  input : pathz = path name for zone files
C          un    = Fortran file unit number
C          zn    = zone number = 326 to 900
C  output: fnzone= name of zone file opened here
C
C  - construct file name from pathz and zn
C  - check existence of file
C  - open file if available (direct access, binary)

      IMPLICIT NONE
      CHARACTER*(*) pathz, fnzone
      INTEGER       un,zn, jp
      LOGICAL       ifex
      CHARACTER*80  answer

      IF (zn.LT.326.OR.zn.GT.900) THEN
        WRITE (*,'(a,i5)') '<open_zfile> invalid zone number = ',zn
        STOP
      ENDIF

 51   jp = INDEX (pathz,' ') - 1
      WRITE (fnzone,'(a,a,i3.3)') pathz(1:jp),'z',zn

      INQUIRE (FILE=fnzone,EXIST=ifex)

      IF (ifex) THEN
        OPEN (un,FILE=fnzone,ACCESS='direct',RECL=80)

      ELSE
        WRITE (*,'(/a)') 'can not find the file:'
        WRITE (*,'(a)') fnzone
        WRITE (*,'(a)') 'please enter new path:'
        WRITE (*,'(a)') '(exit with "x")'
        READ (*,'(a)') answer
        IF (answer.NE.' ') pathz = answer
        IF (pathz(1:2).EQ.'x '.OR.pathz(1:2).EQ.'X ') THEN
          STOP
        ELSE
          GOTO 51
        ENDIF
      ENDIF  

      END   ! subr. <open_zfile>

************************************************************************

      SUBROUTINE getistar (un,nsz,bf,eozf,dv,dimc)
C
C  read data of individual star from URAT binary zone file
C  put all items into data vector I*4
C
C  I:  un   = Fortran unit number of zone file (assumed to be open)
C      nsz  = running star number along zone for which data are requested
C      bf   = byte flip requested = true or false
C      dimc = dimension parameter, max.numb. items (columns) on data vector
C  I/O eozf = .TRUE. if end of zone file is encountered
C  O:  dv(dimv) = data items of star (or zero if eozf)
C  
C  141120 NZ adopt for URAT

      IMPLICIT NONE
* variables from argument list
      INTEGER  un,nsz,dimc, dv(dimc)
      LOGICAL  bf, eozf

* data items of zone file
      INTEGER*4 ra,spd, id2
      INTEGER*2 sigs,sigm, epi, mmag,sigp, nit2,niu2
     .         ,pmr,pmd,pme, jmag,ejmag, hmag,ehmag, kmag,ekmag
      INTEGER*2 apmag(5), aperr(5)
      INTEGER*1 nst1,nsu1, nsm1,ref1, ngt1,ngu1, mfm,mfa
     .         ,iccj,icch,icck, phqj,phqh,phqk, no1,mo1

* others
      INTEGER   i,j

* check
      IF (dimc.LT.45) THEN  ! larger is allowed for extra col. later
        WRITE (*,'(a)') 'error <getistar> need dimc >= 45'
        STOP
      ENDIF

* initialize
      DO i=1,dimc
        dv(i) = 0
      ENDDO

* exit condition
      IF (eozf) RETURN

* actual read of data from file
      READ (un,REC=nsz,ERR=91)                    ! 80 byte
     .  ra,spd, sigs,sigm, nst1,nsu1, epi         ! 8 + 4 + 4
     . ,mmag,sigp, nsm1,ref1, nit2,niu2,ngt1,ngu1 ! 4 + 2 + 6
     . ,pmr,pmd,pme, mfm,mfa, id2                 ! 6 + 2 + 4
     . ,jmag,hmag,kmag, ejmag,ehmag,ekmag         ! 6 + 6
     . ,iccj,icch,icck, phqj,phqh,phqk            ! 3 + 3 
     . ,apmag, aperr, no1,mo1                     ! 10 + 10 + 2

* byte flip on I*2 and I*4 items, if needed
      IF (bf) THEN
        CALL flip4 (ra)
        CALL flip4 (spd)
        CALL flip2 (sigs)
        CALL flip2 (sigm)
        CALL flip2 (epi)
        CALL flip2 (mmag)
        CALL flip2 (sigp)
        CALL flip2 (nit2) 
        CALL flip2 (niu2) 
        CALL flip2 (pmr)
        CALL flip2 (pmd)
        CALL flip2 (pme)
        CALL flip4 (id2)
        CALL flip2 (jmag)
        CALL flip2 (hmag)
        CALL flip2 (kmag)
        CALL flip2 (ejmag)
        CALL flip2 (ehmag)
        CALL flip2 (ekmag)
        DO j=1,5
          CALL flip2 (apmag(j))
          CALL flip2 (aperr(j))
        ENDDO
      ENDIF

* put items into data vector
      dv( 1) = ra
      dv( 2) = spd
      dv( 3) = sigs
      dv( 4) = sigm
      dv( 5) = nst1
      dv( 6) = nsu1
      dv( 7) = epi
      dv( 8) = mmag
      dv( 9) = sigp
      dv(10) = nsm1
      dv(11) = ref1
      dv(12) = nit2
      dv(13) = niu2
      dv(14) = ngt1
      dv(15) = ngu1
      dv(16) = pmr
      dv(17) = pmd
      dv(18) = pme
      dv(19) = mfm
      dv(20) = mfa
      dv(21) = id2
      dv(22) = jmag
      dv(23) = hmag
      dv(24) = kmag
      dv(25) = ejmag
      dv(26) = ehmag
      dv(27) = ekmag
      dv(28) = iccj
      dv(29) = icch
      dv(30) = icck
      dv(31) = phqj
      dv(32) = phqh
      dv(33) = phqk
      dv(34) = apmag(1)
      dv(35) = apmag(2)
      dv(36) = apmag(3)
      dv(37) = apmag(4)
      dv(38) = apmag(5)
      dv(39) = aperr(1)
      dv(40) = aperr(2)
      dv(41) = aperr(3)
      dv(42) = aperr(4)
      dv(43) = aperr(5)
      dv(44) = no1
      dv(45) = mo1

      RETURN          ! normal return with valid data

  91  eozf = .TRUE.   ! end of file encountered
      END  ! subr. <getistar>

************************************************************************

      SUBROUTINE xi_byte_flip (x0,zmax,bmax)
C
C input : x0  = array with index
C         zmax= dimension of x0, max. number of zones
C         bmax= dimension of x0, max. number of RA bins each zone
C output: x0  = same with byte flip applied 

      INTEGER zmax,bmax
      INTEGER x0 (zmax,bmax)
      INTEGER zn,j

      DO zn= 1,zmax
      DO j = 1,bmax
        CALL flip4 (x0(zn,j))
      ENDDO
      ENDDO
      WRITE (*,'(a)') 'index array: byte flip done'

      END   ! subr. <xi_byte_flip>

************************************************************************

      SUBROUTINE flip2 (i2)
C
C input:  Integer*2 value i2
C output: same with byte fliped

      IMPLICIT NONE
      INTEGER*2  i2, i2i,i2o
C     BYTE       a(2), b(2)
      INTEGER*1  a(2), b(2)
      EQUIVALENCE (i2i,a)
      EQUIVALENCE (i2o,b)

      i2i = i2
      b(1) = a(2)
      b(2) = a(1)
      i2 = i2o

      END    ! subr. <flip2>

************************************************************************

      SUBROUTINE flip4 (i4)
C
C input:  Integer*4 value i4
C output: same with byte fliped

      IMPLICIT NONE
      INTEGER*4  i4, i4i,i4o
C     BYTE       a(4), b(4)
      INTEGER*1  a(4), b(4)
      EQUIVALENCE (i4i,a)
      EQUIVALENCE (i4o,b)

      i4i = i4
      b(1) = a(4)
      b(2) = a(3)
      b(3) = a(2)
      b(4) = a(1)
      i4 = i4o

      END    ! subr. <flip4>

************************************************************************

      SUBROUTINE valid_range (dv,dmin,dmax)
C
C  restrict given data value (dv) to given range (dmin,dmax)

      IMPLICIT NONE
      REAL*8   dv, dmin,dmax

      IF (dv.LT.dmin) dv = dmin
      IF (dv.GT.dmax) dv = dmax

      END    ! subr. <valid_range>

************************************************************************

      SUBROUTINE get_zone_range (dc1,dc2,zmax, d1m,d2m,z1,z2,nz)
C
C input:  dc1,dc2 = declination range (degree)
C         zmax    = largest zone number available
C output: d1m,d2m = declination range in mas
C         z1, z2  = req. range of zone numbers (0.2 deg steps)
C         nz      = number of zones, or 0 if out of range

      IMPLICIT NONE
      REAL*8  dc1,dc2, dcx
      INTEGER zmax, d1m,d2m, z1,z2, nz

      IF (dc1.LT.-90.0d0.AND.dc2.LT.-90.0d0) THEN
        nz = 0
        z1 = 1
        z2 = 0
        RETURN
      ENDIF

      CALL valid_range (dc1,-90.0d0,90.0d0)
      CALL valid_range (dc2,-90.0d0,90.0d0)

      IF (dc1.GT.dc2) THEN       ! flip range
        dcx = dc1
        dc1 = dc2
        dc2 = dcx
      ENDIF

      d1m = IDNINT (dc1 * 3.6d6) ! declination degree to mas
      d2m = IDNINT (dc2 * 3.6d6)

      z1 = (d1m + 324000000) / 720000 + 1   ! 720" = 0.2 deg
      z2 = (d2m + 323999999) / 720000 + 1

      IF (z2.GT.zmax) z2 = zmax

      IF (z1.GT.zmax) THEN       ! out of available zone range
        z1 = zmax + 1
        nz = 0
      ELSE
        nz = z2 - z1 + 1
      ENDIF

      END   !  subr. <get_zone_range>

************************************************************************

      SUBROUTINE get_ra_range (ra1,ra2, ralo,rahi,i1,i2,nr)
C
C  input:  ra1,ra2   = RA range (hour)
C  output: ralo,rahi = range of RA in mas (1 or 2)
C          i1, i2    = range in index for 1 arcmin bins
C          nr        = number of ranges = 1 or 2
C    2 ranges possible, if ra1 > ra2,
C    then assume cross over 24/0 hour in RA --> 2 ranges
C    (e.g. 23...1 --> 23...24, and  0...1 hour for output)

      IMPLICIT NONE
      REAL*8  ra1,ra2
      INTEGER ralo(2),rahi(2), i1(2),i2(2), nr
      INTEGER r1m,r2m
      REAL*8  rax

      CALL valid_range (ra1, 0.0d0,24.0d0)
      CALL valid_range (ra2, 0.0d0,24.0d0)

      r1m = IDNINT (ra1 * 5.4d7)   ! RA hour to mas
      r2m = IDNINT (ra2 * 5.4d7)

      IF (r1m.LE.r2m) THEN         ! normal case
        nr = 1
        i1(1) =  r1m    / 900000 + 1  ! 900 arcsec = 0.25 deg
        i2(1) = (r2m-1) / 900000 + 1
        i1(2) = 1
        i2(2) = 0
        ralo(1) = r1m
        rahi(1) = r2m
        ralo(2) = 0
        rahi(2) = 0

      ELSE                         ! cross over 24/0
        nr = 2
        i1(1) =  r1m    / 900000 + 1
        i2(1) = 1440               ! = last bin up to 24 hour
        i1(2) =   1
        i2(2) = (r2m-1) / 900000 + 1
        ralo(1) = r1m
        rahi(1) = 1296000000       ! 24 hour in mas
        ralo(2) = 0
        rahi(2) = r2m
      ENDIF

      END   ! subr. <get_ra_range>

************************************************************************

      SUBROUTINE as2hms (ra,dk,crekt,cdekl)
C
C convert R*8 RA, DC (arcsec) to hms, dms strings
C
C 940725 NZ update to CHARACTER*13 to 1/1000 arcsec

      IMPLICIT REAL*8 (A-H,L-Z)

      REAL*8     RA, DK   
      INTEGER*4  IRASTD, IRAMIN, IDKGRD,IDKMIN
      CHARACTER*1  CVZ
      CHARACTER*13 CREKT,CDEKL

      IF (RA.GT.1296000.D0)  RA = RA - 1296000.D0
      IF (RA.LT.      0.D0)  RA = RA + 1296000.D0
      IF (RA.GT.1296000.D0.OR.RA.LT.0.D0)  THEN
        WRITE (90,'(1X//1X,A,F13.3/)')  'RA > 24 hours or < 0 :',RA
        RETURN
      ENDIF

      RASTD = RA/(3600.D0*15.D0)
      IRASTD= IDINT(RASTD)
      RAREST= RASTD-DFLOAT(IRASTD)
      IRAMIN= IDINT(RAREST*60.D0)
      RASEC = RAREST*3600.D0-DFLOAT(IRAMIN)*60.D0

      IF (DABS(RASEC-60.D0).LT.0.001D0)  THEN
        RASEC = 0.D0
        IRAMIN= IRAMIN+1
        IF (IRAMIN.EQ.60)  THEN
          IRAMIN= 0
          IRASTD= IRASTD+1
          IF (IRASTD.EQ.24)  IRASTD= 0
        ENDIF
      ENDIF

      IF (DABS(DK).GT.324000.D0)  THEN
        WRITE (90,'(1X//1X,A,F13.3/)')  'abs (DC)  > 90 deg :',DK
        RETURN
      ENDIF

      DKGRD = DK/3600.D0
      CVZ= '+'

      IF (DK.LT.0.D0)  THEN
        CVZ= '-'
        DKGRD= -DKGRD
      ENDIF

      IDKGRD= IDINT(DKGRD)
      DKREST= DKGRD-DFLOAT(IDKGRD)
      IDKMIN= IDINT(DKREST*60.D0)
      DKSEC = DKREST*3600.D0-DFLOAT(IDKMIN)*60.D0

      IF (DABS(DKSEC-60.D0).LT.0.01D0)  THEN
        DKSEC = 0.D0
        IDKMIN= IDKMIN+1
        IF (IDKMIN.EQ.60)  THEN
          IDKMIN= 0
          IDKGRD= IDKGRD+1
        ENDIF
      ENDIF

      WRITE (CREKT,'(   I2.2,1X,I2.2,1X,F7.4)')
     .    IRASTD,IRAMIN,RASEC
      IF (CREKT(7:7).EQ.' ')  CREKT(7:7)= '0'
      WRITE (CDEKL,'(A1,I2.2,1X,I2.2,1X,F6.3)')
     .    CVZ,IDKGRD,IDKMIN,DKSEC
      IF (CDEKL(8:8).EQ.' ')  CDEKL(8:8)= '0'

      RETURN
      END

************************************************************************

      SUBROUTINE SORTI (A,IS,IW,IV,N,KX,KY)
C   
C  sort of 2-dimensional Integer array 
C
C  input: A    = array
C         IS   = column index to sort by
C         IW   = number of columns to sort 
C         IV,N = lower, upper limit on line numbers to sort 
C         KX,KY= dimension of array A (lines,columns) 
C  output:  A  = same array after sort
 
      DIMENSION A(KX,KY),IL(60),IU(60),T(60),TT(60)
      INTEGER A,T,TT
      M=1
      I=IV
      J=N
      II=I
      GOTO114
  111 IJ=0.01+0.5*(I+J)
      DO115IX=1,IW
  115 T(IX)=A(IJ,IX)
      K=I
      L=J
      IF(A(I,IS).GT.T(IS))THEN
      DO116IX=1,IW
      A(IJ,IX)=A(I,IX)
      A(I,IX)=T(IX)
  116 T(IX)=A(IJ,IX)
      END IF
      IF(A(J,IS).LT.T(IS))THEN
      DO117IX=1,IW
      A(IJ,IX)=A(J,IX)
      A(J,IX)=T(IX)
  117 T(IX)=A(IJ,IX)
      IF(A(I,IS).GT.T(IS))THEN
      DO118IX=1,IW
      A(IJ,IX)=A(I,IX)
      A(I,IX)=T(IX)
  118 T(IX)=A(IJ,IX)
      END IF
      END IF
  112 L=L-1
      IF(A(L,IS).GT.T(IS))GOTO112
      DO119IX=1,IW
  119 TT(IX)=A(L,IX)
  113 K=K+1
      IF(A(K,IS).LT.T(IS))GOTO113
      IF(K.LE.L)THEN
      DO120IX=1,IW
      A(L,IX)=A(K,IX)
  120 A(K,IX)=TT(IX)
      GOTO112
      END IF
      IF((L-I).GT.(J-K))THEN
      IL(M)=I
      IU(M)=L
      I=K
      ELSE
      IL(M)=K
      IU(M)=J
      J=L
      END IF
      M=M+1
  114 IF(J-I.GT.10)GOTO 111
      IF(I.EQ.II)THEN
      IF(I.LT.J)GOTO111
      END IF
      NI=I+1
      DO121IZ=NI,J
      I=IZ
      DO122IX=1,IW
  122 T(IX)=A(IZ,IX)
      K=I-1
      IF(A(K,IS).GT.T(IS))THEN
  123 DO124IX=1,IW
  124 A(K+1,IX)=A(K,IX)
      K=K-1
      IF(A(K,IS).GT.T(IS))GOTO123
      DO125IX=1,IW
  125 A(K+1,IX)=T(IX)
      END IF
  121 CONTINUE
      M=M-1
      IF(M.GE.1)THEN
      I=IL(M)
      J=IU(M)
      GOTO114
      END IF
      RETURN
      END    ! subr. <sorti>

