      PROGRAM v1dump
C
C  gfortran -o v1dump ADM-v1dump.f v1sub.f
C
C  - loop over range of zones
C  - read individual URAT zone file (binary data)
C  - output all data to ASCII file, no header lines
C  - user select column separator character, no empty columns
C  - for explanations of the 45 columns see "readme" file
C
C  141120 NZ adopted from UCAC4 code
C  150119 NZ change to "v", bug fix "238"
C  150126 NZ add URAT star ID zzznnnnnn to output
C  150129 NZ change format item #6 I*3 -> I*4 due to neg.flag (no GSC match)
C  100725 ADM Adapted for NERSC and hardcoded to skip interactive steps
      IMPLICIT NONE
      INTEGER    dimc, dims
      PARAMETER (dimc = 45   ! number of URAT data columns
     .          ,dims = 45)  ! number of separators

      INTEGER      dv(dimc), mi(dimc), ma(dimc), is(dims)
      CHARACTER*1  csc

      INTEGER      i,j,k, jo, uni,uno,zn1,zn2,zn, nst,nsz, idn
      CHARACTER*45 line*223, a1*1, bfc*1
      CHARACTER*100 uratdir, pathi, patho, fnin, fnout
      LOGICAL      eozf, bf

      DATA is /11, 21, 25, 29, 32, 36, 42, 48, 52, 55, 57, 61, 65, 69
     .       , 73, 79, 85, 89, 92, 95,106,112,118,124,129,134,139,141
     .       ,143,145,147,149,151,157,163,169,175,181,186,191,196,201
     .       ,206,210,214/

* defaults
C ADM the input/output directories.
      CALL get_environment_variable("URAT_DIR", uratdir)
      pathi = TRIM(TRIM(uratdir) // "/binary/")
      patho = TRIM(TRIM(uratdir) // "/csv/")
C ADM the file RANGE to convert.
      zn1 = 326
      zn2 = 900
C ADM byte-flip or not (big or little endian).
      bfc = 'N'
C ADM the delimiter for the output files.
      csc = ','

      IF (bfc.EQ.'Y'.OR.bfc.EQ.'y') THEN
        bf = .TRUE.
      ELSE
        bf = .FALSE.
      ENDIF

* prepare
      jo = INDEX (patho,' ') - 2
      uni= 11
      uno= 20
      nst= 0

* loop zone files
      DO zn = zn1,zn2
        WRITE (fnout,'(a,a,i3.3,a)') patho(1:jo),'/z',zn,'.csv'
        CALL open_zfile (pathi,uni,zn,fnin)
        OPEN (uno,FILE=fnout)
        eozf = .FALSE.

        WRITE (*,'(/a,a)') 'begin read file = ',fnin
        WRITE (*,'( a,a)') '... output to   = ',fnout

        DO nsz = 1,999000
          CALL getistar (uni,nsz,bf,eozf,dv,dimc)
          IF (eozf) GOTO 91

          idn = zn * 1000000 + nsz   ! official star ID number

          WRITE (line,'(2i10,2i4,i3,i4,i6,i6,i4,i3,i2,4i4,2i6,i4
     .                 ,2i3,i11,3i6,3i5,6i2,5i6,5i5,2i4,i10.9)')
     .      (dv(j),j=1,dimc), idn

          IF (csc.NE.' ') THEN
            DO j=1,dims
              k = is(j)
              line(k:k) = csc
            ENDDO
          ENDIF

          WRITE (uno,'(a)') line
        ENDDO   ! loop stars on individ. zone files

  91    CLOSE (uni)
        CLOSE (uno)
        nsz = nsz - 1
        nst = nst + nsz
        WRITE (*,'(a,i7,i10)') 'numb.stars/zone, total = ',nsz,nst
      ENDDO     ! loop all zones

      END  ! main <v1dump>
