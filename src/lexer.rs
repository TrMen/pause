#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignOp {
    Equal,
    PlusEqual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Plus,
    Minus,
    EqualEqual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionDesignator {
    Past,
    Current,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Number,
    String,
    True,
    False,
    BinaryOp(BinaryOp),
    AssignOp(AssignOp),
    ExecutionDesignator(ExecutionDesignator),
    Identifier,
    SmallArrow,
    Procedure,
    Assertion,
    Function,
    SemiColon,
    DotDot,
    Struct,
    Dot,
    Slash,
    Colon,
    Comma,
    Bang,
    LeftParen,
    DoubleQuote,
    RightParen,
    LeftBrace,
    RightBrace,
    LeftBracket,
    RightBracket,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub lexeme: String,
    // Where in the file this is (byte indices)
    start: usize,
    end: usize,
}

impl Token {
    pub fn error_string(&self) -> String {
        // TODO: Calculate line and col from source file taken as param
        format!("{:?}({})", self.kind, self.lexeme)
    }
}

pub struct Lexer<'a> {
    input: &'a [u8],
    span_start: usize,
    index: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            span_start: 0,
            index: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.index).copied()
    }

    fn make_token(&self, kind: TokenKind) -> Token {
        Token {
            kind,
            lexeme: String::from_utf8(self.input[self.span_start..self.index].into()).unwrap(),
            start: self.span_start,
            end: self.index,
        }
    }

    fn advance(&mut self) -> Option<u8> {
        let c = self.peek();

        if self.index < self.input.len() {
            self.index += 1;
        }

        c
    }

    fn advance_if_next_is(&mut self, expected: u8) -> Option<u8> {
        let next = self.peek()?;

        if next == expected {
            self.advance();
            Some(next)
        } else {
            None
        }
    }

    fn next_token(&mut self) -> Option<Token> {
        self.skip_whitespace();

        self.span_start = self.index;

        let c = self.advance()?;

        let kind = match c {
            b'(' => TokenKind::LeftParen,
            b')' => TokenKind::RightParen,
            b'{' => TokenKind::LeftBrace,
            b'}' => TokenKind::RightBrace,
            b',' => TokenKind::Comma,
            b'+' => match self.advance_if_next_is(b'=') {
                Some(_) => TokenKind::AssignOp(AssignOp::PlusEqual),
                None => TokenKind::BinaryOp(BinaryOp::Plus),
            },
            b'-' => match self.advance_if_next_is(b'>') {
                Some(_) => TokenKind::SmallArrow,
                None => TokenKind::BinaryOp(BinaryOp::Minus),
            },
            b';' => TokenKind::SemiColon,
            b':' => TokenKind::Colon,
            b'!' => TokenKind::Bang,
            b'.' => match self.advance_if_next_is(b'.') {
                Some(_) => TokenKind::DotDot,
                None => TokenKind::Dot,
            },
            b'"' => TokenKind::DoubleQuote,
            b'=' => match self.advance_if_next_is(b'=') {
                Some(_) => TokenKind::BinaryOp(BinaryOp::EqualEqual),
                None => TokenKind::AssignOp(AssignOp::Equal),
            },
            b'/' => TokenKind::Slash,
            b'c' => {
                return Some(self.keyword_or_identifier(
                    "urrent",
                    TokenKind::ExecutionDesignator(ExecutionDesignator::Current),
                ))
            }
            b'[' => TokenKind::LeftBracket,
            b']' => TokenKind::RightBracket,
            b't' => return Some(self.keyword_or_identifier("rue", TokenKind::True)),
            b'f' => match self.advance()? {
                b'a' => return Some(self.keyword_or_identifier("lse", TokenKind::False)),
                b'u' => return Some(self.keyword_or_identifier("nction", TokenKind::Function)),
                _ => return Some(self.identifier()),
            },
            b'a' => return Some(self.keyword_or_identifier("ssertion", TokenKind::Assertion)),
            b'p' => match self.advance()? {
                b'r' => return Some(self.keyword_or_identifier("ocedure", TokenKind::Procedure)),
                b'a' => {
                    return Some(self.keyword_or_identifier(
                        "st",
                        TokenKind::ExecutionDesignator(ExecutionDesignator::Past),
                    ))
                }
                _ => return Some(self.identifier()),
            },
            b's' => return Some(self.keyword_or_identifier("truct", TokenKind::Struct)),
            b'0'..=b'9' => return Some(self.number()),
            _ if c.is_ascii_alphanumeric() => return Some(self.identifier()),
            _ => todo!("'{}'", (c as char)),
        };

        Some(self.make_token(kind))
    }

    fn number(&mut self) -> Token {
        while let Some(b'0'..=b'9') = self.peek() {
            self.advance();
        }

        self.make_token(TokenKind::Number)
    }

    fn identifier(&mut self) -> Token {
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == b'_' {
                self.advance();
            } else {
                return self.make_token(TokenKind::Identifier);
            }
        }

        self.make_token(TokenKind::Identifier)
    }

    fn keyword_or_identifier(&mut self, rest: &str, kind: TokenKind) -> Token {
        for byte in rest.as_bytes() {
            if self.peek() != Some(*byte) {
                return self.identifier();
            } else {
                self.advance();
            }
        }

        self.make_token(kind)
    }

    fn peek_nth(&self, n: usize) -> Option<u8> {
        self.input.get(self.index + n).copied()
    }

    fn skip_whitespace(&mut self) {
        while let Some(c @ (b' ' | b'\t' | b'\r' | b'\n' | b'/')) = self.peek() {
            if c == b'/' {
                if let Some(b'/') = self.peek_nth(1) {
                    // Skip the slashes
                    self.advance();
                    self.advance();

                    while self.advance().map_or(false, |c| c != b'\n') {
                        // Skip the rest of the line
                    }
                } else {
                    // Don't skip the slash if only one is there.
                    return;
                }
            } else {
                // In an else block to make sure we don't advance one past the /n at the end of
                // comments
                self.advance();
            }
        }
    }

    pub fn lex(input: &'a [u8]) -> Vec<Token> {
        let mut lexer = Self::new(input);

        let mut tokens = Vec::new();

        while let Some(token) = lexer.next_token() {
            tokens.push(token);
        }

        tokens
    }
}
