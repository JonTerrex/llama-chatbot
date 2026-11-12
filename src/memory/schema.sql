CREATE TABLE IF NOT EXISTS conversations (
    user_id          text        NOT NULL,
    conversation_id  text        NOT NULL,
    next_seq         integer     NOT NULL DEFAULT 1,
    created_at       timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, conversation_id),
    CHECK (user_id         ~ '^[A-Za-z0-9_-]{1,64}$'),
    CHECK (conversation_id ~ '^[A-Za-z0-9_-]{1,64}$')
);

CREATE TABLE IF NOT EXISTS messages (
    user_id          text        NOT NULL,
    conversation_id  text        NOT NULL,
    seq              integer     NOT NULL,
    role             text        NOT NULL CHECK (role IN ('user', 'assistant')),
    content          text        NOT NULL,
    turn_id          uuid        NOT NULL,
    created_at       timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, conversation_id, seq),
    FOREIGN KEY (user_id, conversation_id)
        REFERENCES conversations ON DELETE CASCADE,
    UNIQUE (user_id, conversation_id, turn_id, role)
);
