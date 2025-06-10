WITH albumin_results AS (
    SELECT DISTINCT ON (le.hadm_id) le.hadm_id, 
           CASE
               WHEN le.itemid = 53085 THEN ROUND(le.valuenum::NUMERIC / 1000, 2)
               ELSE ROUND(le.valuenum::NUMERIC, 2)
           END AS albumin_value,
           le.charttime
    FROM mimiciv_hosp.labevents le
    JOIN mimiciv_icu.icustays icu ON le.hadm_id = icu.hadm_id
    JOIN saaki3 s ON s.hadm_id = le.hadm_id
    WHERE le.itemid IN (50862, 53085)
      AND le.charttime BETWEEN TO_TIMESTAMP(s.icu_intime, 'DD/MM/YYYY HH24:MI:SS') - INTERVAL '72 hours'
                           AND TO_TIMESTAMP(s.icu_intime, 'DD/MM/YYYY HH24:MI:SS') + INTERVAL '72 hours'
    ORDER BY le.hadm_id, le.charttime DESC, 
             CASE 
                 WHEN le.itemid = 50862 THEN 1
                 ELSE 2
             END
)
UPDATE saaki3
SET albumin = albumin_results.albumin_value
FROM albumin_results
WHERE saaki3.hadm_id = albumin_results.hadm_id;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki' AND column_name = 'albumin') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN albumin NUMERIC;
    END IF;
END $$;

WITH albumin_results AS (
    SELECT DISTINCT ON (le.hadm_id) 
           le.hadm_id,
           CASE
               WHEN le.itemid = 53085 THEN ROUND(le.valuenum::NUMERIC / 1000, 2)
               ELSE ROUND(le.valuenum::NUMERIC, 2)
           END AS albumin_value,
           le.charttime
    FROM mimiciv_hosp.labevents le
    JOIN mimiciv_icu.icustays icu ON le.hadm_id = icu.hadm_id
    JOIN saaki s ON s.hadm_id = le.hadm_id
    WHERE le.itemid IN (50862, 53085)
      AND le.charttime >= icu.intime
    ORDER BY le.hadm_id, le.charttime ASC,
             CASE 
                 WHEN le.itemid = 50862 THEN 1
                 ELSE 2
             END
)
UPDATE saaki
SET albumin = albumin_results.albumin_value
FROM albumin_results
WHERE saaki.hadm_id = albumin_results.hadm_id;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'saaki3' 
          AND column_name = 'arrhythmia'
    ) THEN
        ALTER TABLE public.ardsbpri ADD COLUMN arrhythmia INTEGER;
    END IF;

    UPDATE public.ardsbpri s
    SET arrhythmia = CASE
        WHEN EXISTS (
            SELECT 1
            FROM mimiciv_hosp.diagnoses_icd d
            JOIN mimiciv_hosp.d_icd_diagnoses dd 
              ON d.icd_code = dd.icd_code
            WHERE d.hadm_id = s.hadm_id
              AND (
                  dd.icd_code LIKE '427%' 
                  OR dd.icd_code LIKE 'I49%'
                  OR dd.long_title ILIKE '%arrhythmia%'
              )
        ) THEN 1
        ELSE 0
    END;
END $$;

WITH converted_height AS (
    SELECT saaki.hadm_id,
           CAST(saaki.weight AS numeric) AS weight_numeric,
           CAST(regexp_replace(saaki.height, '[^0-9.]', '', 'g') AS numeric) AS height_numeric
    FROM saaki
    WHERE saaki.height IS NOT NULL
      AND regexp_replace(saaki.height, '[^0-9.]', '', 'g') ~ '^[0-9]+(\.[0-9]*)?$'
)
UPDATE saaki
SET bmi = ROUND(converted_height.weight_numeric / (POWER((converted_height.height_numeric / 100.0), 2)), 4)
FROM converted_height
WHERE saaki.hadm_id = converted_height.hadm_id
  AND converted_height.weight_numeric IS NOT NULL
  AND converted_height.height_numeric IS NOT NULL
  AND converted_height.height_numeric > 0;

ALTER TABLE simi
DROP COLUMN IF EXISTS bpri;

ALTER TABLE simi
ADD COLUMN bpri NUMERIC(14, 6);

UPDATE simi
SET bpri = CASE 
              WHEN vis IS NOT NULL AND vis > 0 THEN ROUND(CAST(map AS NUMERIC) / CAST(vis AS NUMERIC), 6)
              ELSE NULL
           END;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes 
        WHERE tablename = 'chartevents' AND indexname = 'idx_chartevents_hadm_itemid_charttime'
    ) THEN
        CREATE INDEX idx_chartevents_hadm_itemid_charttime 
        ON mimiciv_icu.chartevents (hadm_id, itemid, charttime);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes 
        WHERE tablename = 'icustays' AND indexname = 'idx_icustays_hadm_intime'
    ) THEN
        CREATE INDEX idx_icustays_hadm_intime 
        ON mimiciv_icu.icustays (hadm_id, intime);
    END IF;
END $$;

UPDATE public.ardsbpri t
SET Calcium = (
    SELECT ROUND(CAST(ce.valuenum AS NUMERIC), 2)
    FROM mimiciv_icu.chartevents ce
    JOIN mimiciv_icu.icustays icu ON icu.hadm_id = ce.hadm_id
    WHERE icu.hadm_id = CAST(t.hadm_id AS INTEGER)
      AND ce.itemid IN (225625, 50893)
      AND ce.charttime > icu.intime
    ORDER BY ce.charttime ASC
    LIMIT 1
);

UPDATE public.ardsbpri t
SET Glucose = (
    SELECT ROUND(CAST(ce.valuenum AS NUMERIC), 2)
    FROM mimiciv_icu.chartevents ce
    JOIN mimiciv_icu.icustays icu ON icu.hadm_id = ce.hadm_id
    WHERE icu.hadm_id = CAST(t.hadm_id AS INTEGER)
      AND ce.itemid IN (226537, 50931, 220621)
      AND ce.charttime > icu.intime
    ORDER BY ce.charttime ASC
    LIMIT 1
);

UPDATE public.ardsbpri t
SET Bun = (
    SELECT ROUND(CAST(ce.valuenum AS NUMERIC), 2)
    FROM mimiciv_icu.chartevents ce
    JOIN mimiciv_icu.icustays icu ON icu.hadm_id = ce.hadm_id
    WHERE icu.hadm_id = CAST(t.hadm_id AS INTEGER)
      AND ce.itemid = 225624
      AND ce.charttime > icu.intime
    ORDER BY ce.charttime ASC
    LIMIT 1
);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tygaki' AND column_name = 'myocardial_infarct') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN myocardial_infarct INTEGER;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tygaki' AND column_name = 'cerebrovascular_disease') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN cerebrovascular_disease INTEGER;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tygaki' AND column_name = 'chronic_pulmonary_disease') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN chronic_pulmonary_disease INTEGER;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tygaki' AND column_name = 'peptic_ulcer_disease') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN peptic_ulcer_disease INTEGER;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tygaki' AND column_name = 'rheumatic_disease') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN rheumatic_disease INTEGER;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tygaki' AND column_name = 'diabetes') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN diabetes INTEGER;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tygaki' AND column_name = 'paraplegia') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN paraplegia INTEGER;
    END IF;

    IF NOT EXISTS (SELECT 极光1 FROM information_schema.columns WHERE table_name = 'tygaki' AND column_name = 'malignant_cancer') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN malignant_cancer INTEGER;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tygaki' AND column_name = 'metastatic_solid_tumor') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN metastatic_solid_tumor INTEGER;
    END极光 IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tygaki' AND column_name = 'aids') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN aids INTEGER;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tygaki' AND column_name = 'charlson_comorbidity_index') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN charlson_comorbidity_index INTEGER;
    END IF;

    UPDATE public.ardsbpri t
    SET myocardial_infarct = CASE WHEN c.myocardial_infarct = 1 THEN 1 ELSE 0 END,
        cerebrovascular_disease = CASE WHEN c.cerebrovascular_disease = 1 THEN 1 ELSE 0 END,
        chronic_pulmonary_disease = CASE WHEN c.chronic_pulmonary_disease = 1 THEN 1 ELSE 0 END,
        peptic_ulcer_disease = CASE WHEN c.peptic_ulcer_disease = 1 THEN 1 ELSE 0 END,
        rheumatic_disease = CASE WHEN c.rheumatic_disease = 1 THEN 1 ELSE 0 END,
        diabetes = CASE WHEN c.diabetes_without_cc = 1 OR c.diabetes_with_cc = 1 THEN 1 ELSE 0 END,
        paraplegia = CASE WHEN c.paraplegia = 1 THEN 1 ELSE 0 END,
        malignant_cancer = CASE WHEN c.malignant_cancer = 1 THEN 1 ELSE 0 END,
        metastatic_solid_tumor = CASE WHEN c.metastatic_solid_tumor = 1 THEN 1 ELSE 0 END,
        aids = CASE WHEN c.aids = 1 THEN 1 ELSE 0 END,
        charlson_comorbidity_index = c.charlson_comorbidity_index
    FROM mimiciv_derived.charlson c
    WHERE t.hadm_id = c.hadm_id;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki3' AND column_name = 'Cirrhosis') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN Cirrhosis INTEGER;
    END IF;

    UPDATE public.ardsbpri s
    SET Cirrhosis = CASE
        WHEN EXISTS (
            SELECT 1
            FROM mimiciv_hosp.diagnoses_ic极光d d
            JOIN mimiciv_hosp.d_icd_diagnoses dd ON d.icd_code = dd.icd_code
            WHERE d.hadm_id = s.hadm_id
              AND (dd.icd_code LIKE '571.5%'
              OR dd.icd_code LIKE 'K74%'
              OR dd.long_title ILIKE '%cirrhosis%')
        ) THEN 1
        ELSE 0
    END;
END $$;

ALTER TABLE saaki
ADD COLUMN IF NOT EXISTS crrt INTEGER;

UPDATE saaki
SET crrt = CASE 
              WHEN EXISTS (
                  SELECT 1
                  FROM mimiciv_derived.rrt r
                  JOIN mimiciv_icu.icustays icu
                      ON r.stay_id = icu.stay_id
                  WHERE r.stay_id = saaki.stay_id
                    AND r.charttime BETWEEN icu.intime AND icu.outtime
              ) THEN 1
              ELSE 0
           END;

WITH esrd_patients AS (
    SELECT DISTINCT subject_id 
    FROM mimiciv_hosp.diagnoses_icd
    WHERE (icd_version = 9 AND icd_code = '5856')
       OR (icd_version = 10 AND icd_code = 'N186')
)
DELETE FROM saaki
WHERE CAST(subject_id AS INTEGER) IN (SELECT subject_id FROM esrd_patients);

WITH short_icu_stay_patients AS (
    SELECT stay_id
    FROM mimiciv_icu.icustays
    WHERE los < 1
)
DELETE FROM saaki
WHERE CAST(stay_id AS INTEGER) IN (SELECT stay_id FROM short_icu_stay_patients);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tygaki' AND column_name = 'gcs') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN gcs INTEGER;
    END IF;

    UPDATE public.ardsbpri
    SET gcs = NULL;

    UPDATE public.ardsbpri t
    SET gcs = (
        SELECT fdg.gcs_min
        FROM mimiciv_derived.first_day_gcs fdg
        WHERE fdg.stay_id = CAST(t.stay_id AS INTEGER)
        LIMIT 1
    );
END $$;

CREATE INDEX IF NOT EXISTS idx_admissions_hadm_id 
ON mimiciv_hosp.admissions (hadm_id);

CREATE INDEX IF NOT EXISTS idx_patients_subject_id 
ON mimiciv_hosp.patients (subject_id);

ALTER TABLE public.ardsbpri ADD COLUMN IF NOT EXISTS gender INTEGER;

UPDATE public.ardsbpri
SET gender = (
    SELECT CASE
        WHEN p.gender = 'M' THEN 1
        WHEN p.gender = 'F' THEN 0
        ELSE NULL
    END
    FROM mimiciv_hosp.patients p
    JOIN mimiciv_hosp.admissions a ON a.subject_id = p.subject_id
    WHERE a.hadm_id = tygaki.hadm_id
);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'tygaki' AND column_name = 'glucose') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN glucose NUMERIC;
    END IF;
END $$;

WITH glucose_results AS (
    SELECT DISTINCT ON (le.hadm_id) le.hadm_id,
           ROUND(le.valuenum::NUMERIC, 2) AS glucose_value,
           le.charttime,
           CASE
               WHEN le.charttime >= icu.intime THEN 1
               ELSE 2
           END AS timing_priority
    FROM mimiciv_hosp.labevents le
    JOIN mimiciv_icu.icustays icu ON le.hadm_id = icu.hadm_id
    JOIN tygaki t ON t.hadm_id = le.hadm_id
    WHERE le.itemid IN (50809, 50931, 52569)
      AND le.charttime BETWEEN icu.intime - INTERVAL '72 hours'
                           AND icu.intime + INTERVAL '72 hours'
    ORDER BY le.hadm_id, timing_priority, le.charttime DESC
)
UPDATE tygaki
SET glucose = glucose_results.glucose_value
FROM glucose_results
WHERE tygaki.hadm_id = glucose_results.hadm_id;

UPDATE public.ardsbpri t
SET Hb = (
    SELECT ROUND(CAST(ce.valuenum AS NUMERIC), 2)
    FROM mimiciv_icu.chartevents ce
    JOIN mimiciv_icu.icustays icu ON icu.hadm_id = ce.hadm_id
    WHERE icu.hadm_id = CAST(t.hadm_id AS INTEGER)
      AND ce.itemid IN (51222, 220228, 50811)
      AND ce.charttime > icu.intime
    ORDER BY ce.charttime ASC
    LIMIT 1
);

UPDATE public.ardsbpri t
SET RBC = (
    SELECT ROUND(CAST(ce.valuenum AS NUMERIC), 2)
    FROM mimiciv_icu.chartevents ce
    JOIN mimiciv_icu.icustays icu ON icu.hadm_id = ce.hadm_id
    WHERE icu.hadm_id = CAST(t.hadm_id AS INTEGER)
      AND ce.itemid IN (52170, 51277)
      AND ce.charttime > icu.intime
    ORDER BY ce.charttime ASC
    LIMIT 1
);

UPDATE public.ardsbpri t
SET RDW = (
    SELECT ROUND(CAST(ce.valuenum AS NUMERIC), 2)
    FROM mimiciv_icu.chartevents ce
    JOIN mimiciv_icu.icustays icu ON icu.hadm_id = ce.hadm_id
    WHERE icu.hadm_id = CAST(t.hadm_id AS INTEGER)
      AND ce.itemid IN (51277, 52172)
      AND ce.charttime > icu.intime
    ORDER BY ce.charttime ASC
    LIMIT 1
);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki3' AND column_name = 'height') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN height NUMERIC;
    END IF;

    UPDATE public.ardsbpri s
    SET height = fdh.height
    FROM mimiciv_derived.first_day_height fdh
    WHERE CAST(s.stay_id AS INTEGER) = fdh.stay_id;
END $$;

DO $$
BEGIN
    IF NOT EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'sepsis3' AND column_name = 'age') THEN
        ALTER TABLE public.sepsis3 ADD COLUMN age NUMERIC;
    END IF;

    IF NOT EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'sepsis3' AND column_name = 'icu_intime') THEN
        ALTER TABLE public.sepsis3 ADD COLUMN icu_intime TIMESTAMP;
    END IF;

    IF NOT EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'sepsis3' AND column_name = 'icu_outtime') THEN
        ALTER TABLE public.sepsis3 ADD COLUMN icu_outtime TIMESTAMP;
    END IF;

    IF NOT EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'sepsis3' AND column_name = 'icu_los') THEN
        ALTER TABLE public.sepsis3 ADD COLUMN icu_los NUMERIC;
    END IF;

    IF NOT EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'sepsis3' AND column_name = 'admittime') THEN
        ALTER TABLE public.sepsis3 ADD COLUMN admittime TIMESTAMP;
    END IF;

    WITH patient_ages AS (
        SELECT 
            CAST(subject_id AS VARCHAR) AS subject_id,
            anchor_age AS age
        FROM mimiciv_hosp.patients
        WHERE subject_id IN (SELECT DISTINCT CAST(subject_id AS INTEGER) FROM public.sepsis3)
    )
    UPDATE public.sepsis3
    SET age = pa.age
    FROM patient_ages pa
    WHERE public.sepsis3.subject_id = pa.subject_id;

    WITH icu_stay_info AS (
        SELECT 
            stay_id,
            subject_id,
            hadm_id,
            intime AS icu_intime,
            outtime AS icu_outtime,
            los AS ic极光_los
        FROM mimiciv_icu.icustays
        WHERE CAST(subject_id AS VARCHAR) IN (SELECT DISTINCT subject_id FROM public.sepsis3)
    )
    UPDATE public.sepsis3
    SET icu_intime = icu_stay_info.icu_intime,
        icu_outtime = icu_stay_info.icu_outtime,
        icu_los = icu_stay_info.icu_los
    FROM icu_stay_info
    WHERE CAST(public.sepsis3.subject_id AS INTEGER) = icu_stay_info.subject_id 
      AND CAST(public.sepsis3.stay_id AS INTEGER) = icu_stay_info.stay_id;

    WITH admission_info AS (
        SELECT 
            subject_id,
            admittime
        FROM mimiciv_hosp.admissions
        WHERE CAST(subject_id AS VARCHAR) IN (SELECT DISTINCT subject_id FROM public.sepsis3)
    )
    UPDATE public.sepsis3
    SET admittime = ai.admittime
    FROM admission_info ai
    WHERE CAST(public.sepsis3.subject_id AS INTEGER) = ai.subject_id;
END $$;

ALTER TABLE public.ardsbpri ADD COLUMN IF NOT EXISTS admittime TIMESTAMP;
ALTER TABLE public.ardsbpri ADD COLUMN IF NOT EXISTS dischtime TIMESTAMP;

WITH admission_info AS (
    SELECT 
        adm.hadm_id,
        adm.admittime,
        adm.dischtime
    FROM mimiciv_hosp.admissions adm
    WHERE adm.hadm_id IN (SELECT CAST(hadm_id AS INTEGER) FROM public.ardsbpri)
)
UPDATE public.ardsbpri
SET admittime = ai.admittime,
    dischtime = ai.dischtime
FROM admission_info ai
WHERE CAST(public.ardsbpri.hadm_id AS INTEGER) = ai.hadm_id;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki3' AND column_name = 'hospital_expire_flag') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN hospital_expire_flag INTEGER;
    END IF;

    UPDATE public.ardsbpri s
    SET hospital_expire_flag = a."hospital_expire_flag"
    FROM mimiciv_hosp.admissions a
    WHERE s.hadm_id = a.hadm_id;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'chartevents' AND indexname = 'idx_chartevents_stayid_itemid_charttime') THEN
        CREATE INDEX idx_chartevents_stayid_itemid_charttime 
        ON mimiciv_icu.chartevents (stay_id, itemid, charttime);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'icustays' AND indexname = 'idx_icustays_stayid_intime') THEN
        CREATE INDEX idx_icustays_stayid_intime 
        ON mimiciv_icu.icustays (stay_id, intime);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki3' AND column_name = 'hr') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN hr INTEGER;
    END IF;

    WITH latest_hr AS (
        SELECT 
            vitals.stay_id,
            vitals.valuenum AS hr_value
        FROM mimiciv_icu.chartevents vitals
        JOIN mimiciv_icu.icustays icu ON vitals.stay_id = icu.stay_id
        WHERE vitals.itemid = 220045
          AND vitals.charttime > icu.intime
          AND NOT EXISTS (
              SELECT 1
              FROM mimiciv_icu.chartevents v2
              WHERE v2.stay_id = vitals.stay_id
                AND v2.itemid = vitals.itemid
                AND v2.charttime > icu.intime
                AND v2.charttime < vitals.charttime
          )
    )
    UPDATE public.ardsbpri s
    SET hr = latest_hr.hr_value
    FROM latest_hr
    WHERE s.stay_id = latest_hr.stay_id;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki3' AND column_name = 'hypertension') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN hypertension INTEGER;
    END IF;

    UPDATE public.ardsbpri t
    SET hypertension = CASE
        WHEN EXISTS (
            SELECT 1
            FROM mimiciv_hosp.diagnoses_icd d
            JOIN mimiciv_hosp.d_icd_diagnoses dd ON d.icd_code = dd.icd_code
            WHERE d.hadm_id = t.hadm_id
              AND (dd.icd_code LIKE '401%' OR dd.icd_code LIKE '402%' OR dd.icd_code LIKE '403%' 
                   OR dd.icd_code LIKE '404%' OR dd.icd_code LIKE '405%'
              OR dd.icd_code LIKE 'I10%' OR dd.icd_code LIKE 'I11%' OR dd.icd_code LIKE 'I12%' 
                   OR dd.icd_code LIKE 'I13%' OR dd.icd_code LIKE 'I15%'
              OR dd.long_title ILIKE '%hypertension%')
        ) THEN 1
        ELSE 0
    END;
END $$;

CREATE INDEX IF NOT EXISTS idx_diagnoses_icd_hadm_id_icd_code
ON mimiciv_hosp.diagnoses_icd (hadm_id, icd_code);

CREATE INDEX IF NOT EXISTS idx_d_icd_diagnoses_icd_code_long_title
ON mimiciv_hosp.d_icd_diagnoses (icd_code, long_title);

CREATE INDEX IF NOT EXISTS idx_saaki3_hadm_id
ON public.ardsbpri (hadm_id);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki3' AND column_name = 'INR') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN INR NUMERIC(10, 2);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki3' AND column_name = 'PT') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN PT NUMERIC(10, 2);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki3' AND column_name = 'PTT') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN PTT NUMERIC(10, 2);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_chartevents_hadm_id_itemid
ON mimiciv_icu.chartevents (hadm_id, itemid);

CREATE INDEX IF NOT EXISTS idx_chartevents_charttime
ON mimiciv_icu.chartevents (charttime);

CREATE INDEX IF NOT EXISTS idx_icustays_hadm_id
ON mimiciv_icu.icustays (hadm_id);

CREATE INDEX IF NOT EXISTS idx_saaki3_hadm_id
ON public.ardsbpri (hadm_id);

UPDATE public.ardsbpri t
SET 
    INR = (
        SELECT ROUND(CAST(ce.valuenum AS NUMERIC), 2)
        FROM mimiciv_icu.chartevents ce
        JOIN mimiciv_icu.icustays icu ON icu.hadm_id = ce.hadm_id
        WHERE icu.hadm_id = CAST(t.hadm_id AS INTEGER)
          AND ce.itemid IN (51237, 227467)
          AND ce.charttime > icu.intime
        ORDER BY ce.charttime ASC
        LIMIT 1
    ),
    PT = (
        SELECT ROUND(CAST(ce.valuenum AS NUMERIC), 2)
        FROM mimiciv_icu.chartevents ce
        JOIN mimiciv_icu.icustays icu ON icu.hadm_id = ce.hadm_id
        WHERE icu.hadm_id = CAST(t.hadm_id AS INTEGER)
          AND ce.itemid IN (51274, 227466)
          AND ce.charttime > icu.intime
        ORDER BY ce.charttime ASC
        LIMIT 1
    ),
    PTT = (
        SELECT ROUND(CAST(ce.valuenum AS NUMERIC), 2)
        FROM mimiciv_icu.chartevents ce
        JOIN mimiciv_icu.icustays icu ON icu.hadm_id = ce.hadm_id
        WHERE icu.hadm_id = CAST(t.hadm_id AS INTEGER)
          AND ce.itemid IN (51275, 227466)
          AND ce.charttime > icu.intime
        ORDER BY ce.charttime ASC
        LIMIT 1
    );

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki3' AND column_name = 'Morality28d') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN Morality28d INTEGER;
    END IF;

    UPDATE public.ardsbpri s
    SET Morality28d = CASE
        WHEN a.deathtime IS NOT NULL AND a.deathtime <= a.admittime + INTERVAL '28 days' THEN 1
        ELSE 0
    END
    FROM mimiciv_hosp.admissions a
    WHERE s.hadm_id = a.hadm_id;
END $$;

ALTER TABLE public.ardsbpri 
ADD COLUMN IF NOT EXISTS oasis_score INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS apsiii_score INTEGER DEFAULT 0;

WITH scores AS (
    SELECT 
        t.stay_id,
        COALESCE(o.oasis_score, 0) AS oasis_score,
        COALESCE(a.apsiii_score, 0) AS apsiii_score
    FROM public.ardsbpri t
    LEFT JOIN (
        SELECT 
            CAST(stay_id AS VARCHAR) AS stay_id,
            oasis AS oasis_score
        FROM mimiciv_derived.oasis
    ) o ON t.stay_id = o.stay_id
    LEFT JOIN (
        SELECT 
            CAST(stay_id AS VARCHAR) AS stay_id,
            apsiii AS apsiii_score
        FROM mimiciv_derived.apsiii
    ) a ON t.stay_id = a.stay_id
)
UPDATE public.ardsbpri
SET 
    oasis_score = scores.oasis_score,
    apsiii_score = scores.apsiii_score
FROM scores
WHERE public.ardsbpri.stay_id = scores.stay_id;

UPDATE public.ardsbpri s
SET pH = (
    SELECT ROUND(CAST(ce.valuenum AS NUMERIC), 2)
    FROM mimiciv_icu.chartevents ce
    JOIN mimiciv_icu.icustays icu ON icu.hadm_id = ce.hadm_id
    WHERE icu.hadm_id = CAST(s.hadm_id AS INTEGER)
      AND ce.itemid IN (223830)
      AND ce.charttime > icu.intime
    ORDER BY ce.charttime ASC
    LIMIT 1
);

UPDATE public.ardsbpri s
SET pCO2 = (
    SELECT ROUND(CAST(ce.valuenum AS NUMERIC), 2)
    FROM mimiciv_icu.chartevents ce
    JOIN mimiciv_icu.icustays icu ON icu.hadm_id = ce.hadm_id
    WHERE icu.hadm_id = CAST(s.hadm_id AS INTEGER)
      AND ce.itemid IN (220235)
      AND ce.charttime > icu.intime
    ORDER BY ce.charttime ASC
    LIMIT 1
);

UPDATE public.ardsbpri s
SET pO2 = (
    SELECT ROUND极光(CAST(ce.valuenum AS NUMERIC), 2)
    FROM mimiciv_icu.chartevents ce
    JOIN mimiciv_icu.icustays icu ON icu.hadm_id = ce.hadm_id
    WHERE icu.hadm_id = CAST(s.hadm_id AS INTEGER)
      AND ce.itemid IN (220224)
      AND ce.charttime > icu.intime
    ORDER BY ce.charttime ASC
    LIMIT 1
);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki' AND column_name = 'ph') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN ph NUMERIC;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki' AND column_name = 'pco2') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN pco2 NUMERIC;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki' AND column_name = 'po2') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN po2 NUMERIC;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki' AND column_name = 'lactate') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN lactate NUMERIC;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki' AND column_name = 'base_excess') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN base_excess NUMERIC;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_labevents_hadm_id_itemid
ON mimiciv_hosp.labevents (hadm_id, itemid);

CREATE INDEX IF NOT EXISTS idx_labevents_charttime
ON mimiciv_hosp.labevents (charttime);

CREATE INDEX IF NOT EXISTS idx_icustays_hadm_id
ON mimiciv_icu.icustays (hadm_id);

CREATE INDEX IF NOT EXISTS idx_saaki_hadm_id
ON public.ardsbpri (hadm_id);

WITH abg_results AS (
    SELECT le.hadm_id,
           ROUND(MAX(CASE WHEN le.itemid = 50820 THEN le.valuenum::NUMERIC END), 2) AS ph_value,
           ROUND(MAX(CASE WHEN le.itemid = 50818 THEN le.valuenum::NUMERIC END), 2) AS pco2_value,
           ROUND(MAX(CASE WHEN le.itemid = 50821 THEN le.valuenum::NUMERIC END), 2) AS po2_value,
           ROUND(MAX(CASE WHEN le.itemid = 50813 THEN le.valuenum::NUMERIC END), 2) AS lactate_value,
           ROUND(MAX(CASE WHEN le.itemid = 50802 THEN le.valuenum::NUMERIC END), 2) AS base_excess_value,
           ROW_NUMBER() OVER (PARTITION BY le.hadm_id ORDER BY le.charttime ASC) AS rn
    FROM mimiciv_hosp.labevents le
    JOIN mimiciv_icu.icustays icu ON le.hadm_id = icu.hadm_id
    JOIN saaki s ON s.hadm_id = le.hadm_id
    WHERE le.itemid IN (50820, 50818, 50821, 50813, 50802)
      AND le.charttime >= icu.intime
    GROUP BY le.hadm_id, le.charttime
)
UPDATE public.ardsbpri
SET ph = abg_results.ph_value,
    pco2 = abg_results.pco2_value,
    po2 = abg_results.po2_value,
    lactate = abg_results.lactate_value,
    base_excess = abg_results.base_excess_value
FROM abg_results
WHERE public.ardsbpri.hadm_id = abg_results.hadm_id
  AND abg_results.rn = 1;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki3' AND column_name = 'potassium') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN potassium NUMERIC;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki3' AND column_name = 'hematocrit') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN hematocrit NUMERIC;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki3' AND column_name = 'creatinine') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN creatinine NUMERIC;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki3' AND column_name = 'ck_mb') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN ck_mb NUMERIC;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki3' AND column_name = 'troponin_t') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN troponin_t NUMERIC;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki3' AND column_name = 'lactate') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN lactate NUMERIC;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki3' AND column_name = 'base_excess') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN base_excess NUMERIC;
    END IF;
END $$;

UPDATE public.ardsbpri t
SET potassium = ROUND(CAST((
    SELECT vitals.valuenum
    FROM mimiciv_icu.chartevents vitals
    JOIN mimiciv_icu.icustays icu ON icu.stay_id = CAST(t.stay_id AS INTEGER)
    WHERE vitals.stay_id = CAST(t.stay_id AS INTEGER)
      AND vitals.itemid IN (227442, 220640)
      AND vitals.charttime > icu.intime
    ORDER BY vitals.charttime ASC
    LIMIT 1
) AS NUMERIC), 1);

UPDATE public.ardsbpri t
SET hematocrit = ROUND(CAST((
    SELECT vitals.valuenum
    FROM mimiciv_icu.chartevents vitals
    JOIN mimiciv_icu.icustays icu ON icu.stay_id = CAST(t.stay_id AS INTEGER)
    WHERE vitals.stay_id = CAST(t.stay_id AS INTEGER)
      AND vitals.itemid IN (220545, 226540)
      AND vitals.charttime > icu.intime
    ORDER BY vitals.charttime ASC
    LIMIT 1
) AS NUMERIC), 1);

UPDATE public.ardsbpri t
SET creatinine = ROUND(CAST((
    SELECT vitals.valuenum
    FROM mimiciv_icu.chartevents vitals
    JOIN mimiciv_icu.icustays icu ON icu.stay_id = CAST(t.stay_id AS INTEGER)
    WHERE vitals.stay_id = CAST(t.stay_id AS INTEGER)
      AND vitals.itemid IN (220615)
      AND vitals.charttime > icu.intime
    ORDER BY vitals.charttime ASC
    LIMIT 1
) AS NUMERIC), 1);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki3' AND column_name = 'race') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN race INTEGER;
    END IF;

    UPDATE public.ardsbpri s
    SET race = CASE
           WHEN a.race IN ('BLACK/AFRICAN', 'BLACK/AFRICAN AMERICAN', 'BLACK/CAPE VERDEAN', 'BLACK/CARIBBEAN ISLAND') THEN 1
           WHEN a.race IN ('WHITE', 'WHITE - BRAZILIAN', 'WHITE - EASTERN EUROPEAN', 'WHITE - OTHER EUROPEAN', 'WHITE - RUSSIAN') THEN 2
           WHEN a.race LIKE 'HISPANIC%' OR a.race IN ('AMERICAN INDIAN/ALASKA NATIVE', 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER') THEN 3
           WHEN a.race LIKE 'ASIAN%' THEN 4
           ELSE 5
       END
    FROM mimiciv_hosp.admissions a
    WHERE s.hadm_id = a.hadm_id;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki' AND column_name = 'rr') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN rr INTEGER;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki' AND column_name = 'sbp') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN sbp NUMERIC;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki'极光 AND column_name = 'dbp') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN dbp NUMERIC;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki' AND column_name = 'map') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN map NUMERIC(5, 2);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki' AND column_name = 't') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN t NUMERIC(5, 2);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'map_errors') THEN
        CREATE TABLE map_errors (
            stay_id INTEGER,
            sbp NUMERIC,
            dbp NUMERIC,
            calculated_map NUMERIC,
            error_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    END IF;

    UPDATE public.ardsbpri s
    SET rr = (
        SELECT vitals.valuenum
        FROM mimiciv_icu.chartevents vitals
        JOIN mimiciv_icu.icustays icu ON icu.stay_id = CAST(s.stay_id AS INTEGER)
        WHERE vitals.stay_id = CAST(s.stay_id AS INTEGER)
          AND vitals.itemid IN (220210, 224690)
          AND vitals.charttime BETWEEN icu.intime AND icu.intime + INTERVAL '24 hours'
        ORDER BY vitals.charttime ASC
        LIMIT 1
    );

    UPDATE public.ardsbpri
    SET map = CASE
                 WHEN sbp + 2 * dbp <= 999 THEN ROUND((sbp + 2 * dbp) / 3.0, 2)
                 ELSE NULL
             END
    WHERE sbp IS NOT NULL AND dbp IS NOT NULL;

    UPDATE public.ardsbpri s
    SET t = (
        SELECT vitals.valuenum
        FROM mimiciv_icu.chartevents vitals
        JOIN mimiciv_icu.icustays icu ON icu.stay_id = CAST(s.stay_id AS INTEGER)
        WHERE vitals.stay_id = CAST(s.stay_id AS INTEGER)
          AND vitals.itemid IN (223761, 223762)
          AND vitals.charttime BETWEEN icu.intime AND icu.intime + INTERVAL '24 hours'
        ORDER BY vitals.charttime ASC
        LIMIT 1
    );
END $$;

UPDATE saaki s
SET t = (
    SELECT 
        CASE 
            WHEN vitals.itemid = 223762 THEN (vitals.valuenum * 9/5) + 32
            ELSE vitals.valuenum
        END AS fahrenheit
    FROM mimiciv_icu.chartevents vitals
    JOIN mimiciv_icu.icustays icu ON icu.stay_id = CAST(s.stay_id AS INTEGER)
    WHERE vitals.stay_id = CAST(s.stay_id AS INTEGER)
      AND vitals.itemid IN (223761, 223762)
      AND vitals.charttime BETWEEN icu.intime AND icu.intime + INTERVAL '24 hours'
    ORDER BY vitals.charttime ASC
    LIMIT 1
);

WITH recent_sapsii AS (
    SELECT 
        sapsii.hadm_id,
        sapsii.sapsii AS sapsii_score,
        sapsii.starttime AS score_time,
        ROW_NUMBER() OVER (PARTITION BY sapsii.hadm_id ORDER BY sapsii.starttime ASC) AS rn
    FROM 
        mimiciv_derived.sapsii
    JOIN 
        mimiciv_icu.icustays ON sapsii.hadm_id = icustays.hadm_id
    WHERE 
        sapsii.starttime >= icustays.intime
)
UPDATE 
    public.ardsbpri
SET 
    sapsii = recent_sapsii.sapsii_score
FROM 
    recent_sapsii
WHERE 
    CAST(public.ardsbpri.hadm_id AS INTEGER) = recent_sapsii.hadm_id
    AND recent_sapsii.rn = 1;

SELECT 
    s3.subject_id,
    s3.stay_id,
    icu.hadm_id
FROM mimiciv_derived.sepsis3 s3
JOIN mimiciv_icu.icustays icu
    ON s3.subject_id = icu.subject_id 
AND s3.stay_id = icu.stay_id;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki3' AND column_name = 'sofa_score') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN sofa_score INTEGER;
    END IF;

    UPDATE public.ardsbpri s
    SET sofa_score = (
        SELECT se.sofa_score
        FROM mimiciv_derived.sepsis3 se
        WHERE CAST(se.stay_id AS VARCHAR) = s.stay_id
        LIMIT 1
    );
END $$;

DO $$
BEGIN
    UPDATE public.ardsbpri s
    SET
        RBC = (
            SELECT ROUND(le.valuenum::NUMERIC, 2)
            FROM mimiciv_hosp.labevents le
            JOIN mimiciv_icu.icustays icu ON le.hadm_id = icu.hadm_id
            WHERE le.itemid = 52170
              AND le.charttime > icu.intime
              AND le.hadm_id = s.hadm_id
            ORDER BY le.charttime ASC
            LIMIT 1
        ),
        RDW = (
            SELECT ROUND(le.valuenum::NUMERIC, 2)
            FROM mimiciv_hosp.labevents le
            JOIN mimiciv_icu.icustays icu ON le.hadm_id = icu.hadm_id
            WHERE le.itemid = 51277
              AND le.charttime > icu.intime
              AND le.hadm_id = s.hadm_id
            ORDER BY le.charttime ASC
            LIMIT 1
        ),
        Lactate = (
            SELECT ROUND(le.valuenum::NUMERIC, 2)
            FROM mimiciv_hosp.labevents le
            JOIN mimiciv_icu.icustays icu ON le.hadm_id = icu.hadm_id
            WHERE le.itemid = 50813
              AND le.charttime > icu.intime
              AND le.hadm_id = s.hadm_id
            ORDER BY le.charttime ASC
            LIMIT 1
        ),
        Base_Excess = (
            SELECT ROUND(le.valuenum::NUMERIC, 2)
            FROM mimiciv_hosp.labevents le
            JOIN mimiciv_icu.icustays icu ON le.hadm_id = icu极光.hadm_id
            WHERE le.itemid = 50802
              AND le.charttime > icu.intime
              AND le.hadm_id = s.hadm_id
            ORDER BY le.charttime ASC
            LIMIT 1
        ),
        Bicarbonate = (
            SELECT ROUND(le.valuenum::NUMERIC, 2)
            FROM mimiciv_hosp.labevents le
            JOIN mimiciv_icu.icustays icu ON le.hadm_id = icu.hadm_id
            WHERE le.itemid = 50882
              AND le.charttime > icu.intime
              AND le.hadm_id = s.hadm_id
            ORDER BY le.charttime ASC
            LIMIT 1
        );
END $$;

DO $$
BEGIN
    UPDATE public.ardsbpri t
    SET Ventilation = CASE
        WHEN EXISTS (
            SELECT 1
            FROM mimiciv_derived.ventilation v
            WHERE CAST(v.stay_id AS VARCHAR) = t.stay_id
              AND (v.ventilation_status = 'InvasiveVent' OR v.ventilation_status = 'NonInvasiveVent')
        ) THEN 1
        ELSE 0
    END;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'saaki3' AND column_name = 'weight') THEN
        ALTER TABLE public.ardsbpri ADD COLUMN weight NUMERIC;
    END IF;

    UPDATE public.ardsbpri s
    SET weight = fdw.weight_admit
    FROM mimiciv_derived.first_day_weight fdw
    WHERE CAST(s.stay_id AS INTEGER) = fdw.stay_id;
END $$;

ALTER TABLE public.ardsbpri ADD COLUMN IF NOT EXISTS avg_wbc NUMERIC(10,4);
ALTER TABLE public.ardsbpri ADD COLUMN IF NOT EXISTS avg_basophils_abs NUMERIC(10,4);
ALTER TABLE public.ardsbpri ADD COLUMN IF NOT EXISTS avg_eosinophils_abs NUMERIC(10,4);
ALTER TABLE public.ardsbpri ADD COLUMN IF NOT EXISTS avg_lymphocytes_abs NUMERIC(10,4);
ALTER TABLE public.ardsbpri ADD COLUMN IF NOT EXISTS avg_monocytes_abs NUMERIC(10,4);
ALTER TABLE public.ardsbpri ADD COLUMN IF NOT EXISTS avg_neutrophils_abs NUMERIC(10,4);
ALTER TABLE public.ardsbpri ADD COLUMN IF NOT EXISTS avg_basophils NUMERIC(10,4);
ALTER TABLE public.ardsbpri ADD COLUMN IF NOT EXISTS avg_eosinophils NUMERIC(10,4);
ALTER TABLE public.ardsbpri ADD COLUMN IF NOT EXISTS avg_lymphocytes NUMERIC(10,4);
ALTER TABLE public.ardsbpri ADD COLUMN IF NOT EXISTS avg_monocytes NUMERIC(10,4);
ALTER TABLE public.ardsbpri ADD COLUMN IF NOT EXISTS avg_neutrophils NUMERIC(10,4);
ALTER TABLE public.ardsbpri ADD COLUMN IF NOT EXISTS avg_immature_granulocytes NUMERIC(10,4);
ALTER TABLE public.ardsbpri ADD COLUMN IF NOT EXISTS avg_platelet NUMERIC(10,4);

WITH blood_results AS (
    SELECT
        bd.hadm_id,
        ROUND(AVG(NULLIF(bd.wbc, 0))::NUMERIC, 4) AS avg_wbc,
        ROUND(AVG(NULLIF(bd.basophils_abs, 0))::NUMERIC, 4) AS avg_basophils_abs,
        ROUND(AVG(NULLIF(bd.eosinophils_abs, 0))::NUMERIC, 4) AS avg_eosinophils_abs,
        ROUND(AVG(NULLIF(bd.lymphocytes_abs, 0))::NUMERIC, 4) AS avg_lymphocytes_abs,
        ROUND(AVG(NULLIF(bd.monocytes_abs, 0))::NUMERIC, 4) AS avg_monocytes_abs,
        ROUND(AVG(NULLIF(bd.neutrophils_abs, 0))::NUMERIC, 4) AS avg_neutrophils_abs,
        ROUND(AVG(NULLIF(bd.basophils, 0))::NUMERIC, 4) AS avg_basophils,
        ROUND(AVG(NULLIF(bd.eosinophils, 0))::NUMERIC, 4) AS avg_eosinophils,
        ROUND(AVG(NULLIF(bd.lymphocytes, 0))::NUMERIC, 4) AS avg_lymphocytes,
        ROUND(AVG(NULLIF(bd.monocytes, 0))::NUMERIC, 4) AS avg_monocytes,
        ROUND(AVG(NULLIF(bd.neutrophils, 0))::NUMERIC, 4) AS avg_neutrophils,
        ROUND(AVG(NULLIF(bd.immature_granulocytes, 0))::NUMERIC, 4) AS avg_immature_granulocytes
    FROM mimiciv_derived.blood_differential bd
    JOIN public.ardsbpri s ON bd.hadm_id = CAST(s.hadm_id AS INTEGER)
    WHERE bd.charttime BETWEEN (s.icu_intime - INTERVAL '24 hours') AND (s.icu_intime + INTERVAL '24 hours')
    GROUP BY bd.hadm_id
),
platelet_results AS (
    SELECT
        le.hadm_id,
        ROUND(AVG(NULLIF(le.valuenum, 0))::NUMERIC, 4) AS avg_platelet
    FROM mimiciv_hosp.labevents le
    JOIN public.ardsbpri s ON le.hadm_id = CAST(s.hadm_id AS INTEGER)
    WHERE le.itemid = 51265
      AND le.charttime BETWEEN (s.icu_intime - INTERVAL '24 hours') AND (s.icu_intime + INTERVAL '24 hours')
    GROUP BY le.hadm_id
)
UPDATE public.ardsbpri
SET avg_wbc = br.avg_wbc,
    avg_basophils_abs = br.avg_basophils_abs,
    avg_eosinophils_abs = br.avg_eosinophils_abs,
    avg_lymphocytes_abs = br.avg_lymphocytes_abs,
    avg_monocytes_abs = br.avg_monocytes_abs,
    avg_neutrophils_abs = br.avg_neutrophils_abs,
    avg_basophils = br.avg_basophils,
    avg_eosinophils = br.avg_eosinophils,
    avg_lymphocytes = br.avg_lymphocytes,
    avg_monocytes = br.avg_monocytes,
    avg_neutrophils = br.avg_neutrophils,
    avg_immature_granulocytes = br.avg_immature_granulocytes,
    avg_platelet = pr.avg_platelet
FROM blood_results br
LEFT JOIN platelet_results pr ON br.hadm_id = pr.hadm_id
WHERE CAST(public.ardsbpri.hadm_id AS INTEGER) = br.hadm_id;

CREATE INDEX idx_chartevents_stay_id_itemid
ON mimiciv_icu.chartevents (stay_id, itemid);

CREATE INDEX idx_chartevents_charttime
ON mimiciv_icu.chartevents (charttime);

CREATE INDEX idx_icustays_stay_id
ON mimiciv_icu.icustays (stay_id);

CREATE INDEX idx_saaki3_stay_id
ON public.ardsbpri (stay_id);

ALTER TABLE ardsbpri
ADD COLUMN IF NOT EXISTS alt NUMERIC,
ADD COLUMN IF NOT EXISTS alp NUMERIC,
ADD COLUMN IF NOT EXISTS ast NUMERIC;

DROP TABLE IF EXISTS temp_simi_avg;

CREATE TEMP TABLE temp_simi_avg AS
SELECT
    stay_id,
    (alt_min + alt_max) / 2 AS alt,
    (alp_min + alp_max) / 2 AS alp,
    (ast_min + ast_max) / 2 AS ast
FROM public.ardsbpri
WHERE alt_min IS NOT NULL AND alt_max IS NOT NULL
  AND alp_min IS NOT NULL AND alp_max IS NOT NULL
  AND ast_min IS NOT NULL AND ast_max IS NOT NULL;

UPDATE ardsbpri
SET
    alt = temp_simi_avg.alt,
    alp = temp_simi_avg.alp,
    ast = temp_simi_avg.ast
FROM temp_simi_avg
WHERE ardsbpri.stay_id = temp_simi_avg.stay_id;

DROP TABLE IF EXISTS temp_simi_avg;

ALTER TABLE saaki ADD COLUMN IF NOT EXISTS ARDS INT DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_procedureevents_stay_id ON mimiciv_icu.procedureevents (stay_id);
CREATE INDEX IF NOT EXISTS idx_chartevents_stay_id_itemid ON mimiciv_icu.chartevents (stay_id, itemid);
CREATE INDEX IF NOT EXISTS idx_icustays_stay_id ON mimiciv_icu.icustays (stay_id);
CREATE INDEX IF NOT EXISTS idx_diagnoses_icd_hadm_id ON mimiciv_hosp.diagnoses_icd (hadm_id);

WITH condition_patients AS (
    SELECT ie.hadm_id
    FROM mimiciv_icu.procedureevents pe
    JOIN mimiciv_icu.icustays ie ON pe.stay_id = ie.stay_id
    JOIN mimiciv_icu.chartevents ce ON pe.stay_id = ce.stay_id
    WHERE pe.itemid IN (227194, 227195)
      AND ce.itemid IN (224688, 224689)
      AND ce.valuenum < 300
      AND ce.charttime >= ie.intime
    UNION
    SELECT ie.hadm_id
    FROM mimiciv_icu.chartevents ce
    JOIN mimiciv_icu.icustays ie ON ce.stay_id = ie.stay_id
    WHERE ce.itemid = 224688
      AND ce.valuenum < 300
      AND ce.charttime >= ie.intime
    UNION
    SELECT ie.hadm_id
    FROM mimiciv_icu.chartevents ce
    JOIN mimiciv_icu.icustays ie ON ce.stay_id = ie.stay_id
    WHERE ce.itemid = 224689
      AND ce.valuenum < 300
      AND ce.charttime >= ie.intime
),
ards_confirmed_patients AS (
    SELECT cp.hadm_id
    FROM condition_patients cp
    JOIN mimiciv_hosp.diagnoses_icd di ON cp.hadm_id = di.hadm_id
    WHERE (
        (di.icd_version = 9 AND di.icd_code IN ('51881', '51882', '51884'))
        OR 
        (di.icd_version = 10 AND di.icd_code IN ('J80', 'J9600', 'J9601', 'J9620', 'J9621', 'R09.2'))
    )
)
UPDATE saaki
SET ARDS = CASE
    WHEN hadm_id IN (SELECT hadm_id FROM ards_confirmed_patients) THEN 1
    ELSE 0
END;

CREATE INDEX IF NOT EXISTS idx_diagnoses_icd_hadm_id ON mimiciv_hosp.diagnoses_icd (hadm_id);

WITH excluded_patients AS (
    SELECT DISTINCT di.hadm_id
    FROM mimiciv_hosp.diagnoses_icd di
    WHERE 
        ((di.icd_version = 9 AND di.icd_code LIKE '428%')
         OR (di.icd_version = 10 AND di.icd_code LIKE 'I50%'))
        OR
        ((di.icd_version = 9 AND di.icd_code LIKE '515%')
         OR (di.icd_version = 10 AND di.icd_code LIKE 'J84%'))
        OR
        ((di.icd_version = 9 AND di.icd_code IN ('415.1', '415.19'))
         OR (di.icd_version = 10 AND di.icd_code LIKE 'I26%'))
        OR
        ((di.icd_version = 9 AND di.icd_code LIKE '995.5%' OR di.icd_code LIKE '995.6%' OR di.icd_code LIKE '995.8%')
         OR (di.icd_version = 10 AND di.icd_code LIKE 'T78%'))
        OR
        ((di.icd_version = 9 AND di.icd_code LIKE '162%')
         OR (di.icd_version = 10 AND di.icd_code LIKE 'C34%'))
)
UPDATE saaki
SET ARDS = 0
WHERE ARDS = 1
  AND hadm极光_id IN (SELECT hadm_id FROM excluded_patients);

ALTER TABLE ardsbpri
ADD COLUMN IF NOT EXISTS calcium NUMERIC;

DROP TABLE IF EXISTS temp_simi_avg;

CREATE TEMP TABLE temp_simi_avg AS
SELECT
    stay_id,
    (calcium_min + calcium_max) / 2 AS calcium
FROM public.ardsbpri
WHERE calcium_min IS NOT NULL AND calcium_max IS NOT NULL;

UPDATE ardsbpri
SET
    calcium = temp_simi_avg.calcium
FROM temp_simi_avg
WHERE ardsbpri.stay_id = temp_simi_avg.stay_id;

DROP TABLE IF EXISTS temp_simi_avg;

DELETE FROM public.ardsbpri
WHERE bpri IS NULL;

CREATE INDEX IF NOT EXISTS idx_ardsbpri_stay_id ON ardsbpri (stay_id);
CREATE INDEX IF NOT EXISTS idx_simi_stay_id ON public.ardsbpri (stay_id);
CREATE INDEX IF NOT EXISTS idx_simi_pao2fio2ratio_min_numeric
ON public.ardsbpri (stay_id)
WHERE pao2fio2ratio_min ~ '^[0-9]+(\.[0-9]+)?$';

CREATE TEMP TABLE temp_simi_filtered AS
SELECT 
    stay_id,
    CAST(pao2fio2ratio_min AS NUMERIC) AS pao2fio2ratio_min
FROM public.ardsbpri
WHERE pao2fio2ratio_min ~ '^[0-9]+(\.[0-9]+)?$';

UPDATE ardsbpri
SET pao2fio2ratio = temp_simi_filtered.pao2fio2ratio极光_min
FROM temp_simi_filtered
WHERE ardsbpri.stay_id = temp_simi_filtered.stay_id;

DROP TABLE IF EXISTS temp_simi_filtered;

ALTER TABLE ardsbpri
ADD COLUMN IF NOT EXISTS sodium NUMERIC;

DROP TABLE IF EXISTS temp_simi_avg;

CREATE TEMP TABLE temp_simi_avg AS
SELECT
    stay_id,
    (sodium_min + sodium_max) / 2 AS sodium
FROM public.ardsbpri
WHERE sodium_min IS NOT NULL AND sodium_max IS NOT NULL;

UPDATE ardsbpri
SET
    sodium = temp_simi_avg.sodium
FROM temp_simi_avg
WHERE ardsbpri.stay_id = temp_simi_avg.stay_id;

DROP TABLE IF EXISTS temp_simi_avg;

CREATE INDEX IF NOT EXISTS idx_vasoactive_agent_stay_id ON mimiciv_derived.vasoactive_agent (stay_id);
CREATE INDEX IF NOT EXISTS idx_vasoactive_agent_starttime ON mimiciv_derived.vasoactive_agent (starttime);
CREATE INDEX IF NOT EXISTS idx_icustays_stay_id ON mimiciv_icu.icustays (stay_id);
CREATE INDEX IF NOT EXISTS idx_icustays_intime ON mimiciv_icu.icustays (intime);
CREATE INDEX IF NOT EXISTS idx_saaki_stay_id ON saaki (stay_id);

WITH vis_calculations AS (
    SELECT 
        va.stay_id,
        va.starttime,
        va.endtime,
        COALESCE(va.dopamine, 0) * 1 + 
        COALESCE(va.epinephrine, 0) * 100 + 
        COALESCE(va.norepinephrine, 0) * 100 + 
        COALESCE(va.phenylephrine, 0) / 10 + 
        COALESCE(va.vasopressin, 0) * 10000 + 
        COALESCE(va.dobutamine, 0) * 1 + 
        COALESCE(va.milrinone, 0) * 1 AS vis_value
    FROM mimiciv_derived.vasoactive_agent va
    JOIN mimiciv_icu.icustays icu
        ON va.stay_id = icu.stay_id
    WHERE va.stay_id IN (SELECT stay_id FROM saaki)
      AND va.starttime BETWEEN icu.intime AND icu.intime + INTERVAL '24 hours'
),
max_vis_per_patient AS (
    SELECT DISTINCT ON (stay_id)
        stay_id,
        vis_value AS max_vis,
        starttime AS max_vis_starttime,
        endtime AS max_vis_endtime
    FROM vis_calculations
    ORDER BY stay_id, vis_value DESC, starttime ASC
)
SELECT * INTO TEMP max_vis_temp FROM max_vis_per_patient;

ALTER TABLE saaki
ADD COLUMN IF NOT EXISTS vis FLOAT,
ADD COLUMN IF NOT EXISTS vis_starttime TIMESTAMP,
ADD COLUMN IF NOT EXISTS vis_endtime TIMESTAMP;

UPDATE saaki
SET vis = (
    SELECT max_vis
    FROM max_vis_temp
    WHERE max_vis_temp.stay_id = saaki.stay_id
),
vis_starttime = (
    SELECT max_vis_starttime
    FROM max_vis_temp
    WHERE max_vis_temp.stay_id = saaki.stay_id
),
vis_endtime = (
    SELECT max_vis_endtime
    FROM max_vis_temp
    WHERE max_vis_temp.stay_id = saaki.stay_id
);

DROP TABLE IF EXISTS max_vis_temp;
