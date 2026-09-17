import { useEffect, useRef, useState } from 'react';
import { EditorView, basicSetup } from 'codemirror';
import { json } from '@codemirror/lang-json';
import { oneDark } from '@codemirror/theme-one-dark';
import { EditorState } from '@codemirror/state';
import { KaraokeJSON } from '../utils/transcription/karaokeJSON';

interface JsonEditorProps {
  value: KaraokeJSON;
  onChange: (value: KaraokeJSON) => void;
}

export function JsonEditor({ value, onChange }: JsonEditorProps) {
  const editorRef = useRef<HTMLDivElement>(null);
  const viewRef = useRef<EditorView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isValid, setIsValid] = useState(true);

  useEffect(() => {
    if (!editorRef.current) return;

    const startState = EditorState.create({
      doc: JSON.stringify(value, null, 2),
      extensions: [
        basicSetup,
        json(),
        oneDark,
        EditorView.updateListener.of((update) => {
          if (update.docChanged) {
            const text = update.state.doc.toString();
            try {
              const parsed = JSON.parse(text);
              setError(null);
              setIsValid(true);
              onChange(parsed);
            } catch (e) {
              setError(e instanceof Error ? e.message : 'Invalid JSON');
              setIsValid(false);
            }
          }
        }),
        EditorView.theme({
          '&': {
            height: '500px',
            fontSize: '14px',
          },
          '.cm-scroller': {
            overflow: 'auto',
          },
        }),
      ],
    });

    const view = new EditorView({
      state: startState,
      parent: editorRef.current,
    });

    viewRef.current = view;

    return () => {
      view.destroy();
    };
  }, []);

  // Обновляем содержимое редактора при изменении value извне
  useEffect(() => {
    if (viewRef.current) {
      const currentText = viewRef.current.state.doc.toString();
      const newText = JSON.stringify(value, null, 2);
      
      if (currentText !== newText) {
        viewRef.current.dispatch({
          changes: {
            from: 0,
            to: viewRef.current.state.doc.length,
            insert: newText,
          },
        });
      }
    }
  }, [value]);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">📝 JSON Редактор</h3>
        {isValid ? (
          <span className="text-sm text-green-400">✓ Валидный JSON</span>
        ) : (
          <span className="text-sm text-red-400">✗ Ошибка синтаксиса</span>
        )}
      </div>
      
      {error && (
        <div className="bg-red-500/20 border border-red-500/30 rounded-lg p-3 text-sm text-red-300">
          <strong>Ошибка:</strong> {error}
        </div>
      )}
      
      <div
        ref={editorRef}
        className="border border-gray-700 rounded-lg overflow-hidden"
      />
      
      <div className="text-xs text-gray-500">
        💡 Редактируйте JSON напрямую. Изменения применяются автоматически при валидном синтаксисе.
      </div>
    </div>
  );
}
