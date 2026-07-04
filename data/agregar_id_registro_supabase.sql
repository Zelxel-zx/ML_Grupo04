BEGIN TRANSACTION;

ALTER TABLE "pacientes"
ADD COLUMN IF NOT EXISTS "id_registro" BIGSERIAL;

ALTER TABLE "inventario"
ADD COLUMN IF NOT EXISTS "id_registro" BIGSERIAL;

UPDATE "pacientes"
SET "id_registro" = nextval(pg_get_serial_sequence('"pacientes"', 'id_registro'))
WHERE "id_registro" IS NULL;

UPDATE "inventario"
SET "id_registro" = nextval(pg_get_serial_sequence('"inventario"', 'id_registro'))
WHERE "id_registro" IS NULL;

ALTER TABLE "pacientes"
ALTER COLUMN "id_registro" SET NOT NULL;

ALTER TABLE "inventario"
ALTER COLUMN "id_registro" SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'pacientes_id_registro_key'
    ) THEN
        ALTER TABLE "pacientes"
        ADD CONSTRAINT "pacientes_id_registro_key" UNIQUE ("id_registro");
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'inventario_id_registro_key'
    ) THEN
        ALTER TABLE "inventario"
        ADD CONSTRAINT "inventario_id_registro_key" UNIQUE ("id_registro");
    END IF;
END $$;

COMMIT;
