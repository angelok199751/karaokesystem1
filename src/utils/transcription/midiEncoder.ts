/**
 * MIDI file encoder
 */
import { TranscriptionResult, frameToTime } from './postprocess';

export function generateMidiFile(result: TranscriptionResult): Uint8Array {
  const { notes, sampleRate, hopSize } = result;
  
  // MIDI constants
  const TICKS_PER_QUARTER = 480;
  const TEMPO = 500000; // microseconds per quarter note (120 BPM)
  
  // Convert notes to MIDI events
  const events: MidiEvent[] = [];
  
  for (const note of notes) {
    const startTime = frameToTime(note.startFrame, hopSize, sampleRate);
    const endTime = frameToTime(note.endFrame, hopSize, sampleRate);
    
    const startTick = Math.floor((startTime / (TEMPO / 1000000)) * TICKS_PER_QUARTER);
    const endTick = Math.floor((endTime / (TEMPO / 1000000)) * TICKS_PER_QUARTER);
    
    events.push({
      tick: startTick,
      type: 'noteOn',
      note: note.pitch,
      velocity: note.amplitude
    });
    
    events.push({
      tick: endTick,
      type: 'noteOff',
      note: note.pitch,
      velocity: 0
    });
  }
  
  // Sort events by tick
  events.sort((a, b) => a.tick - b.tick);
  
  // Encode MIDI file
  return encodeMidi(events, TICKS_PER_QUARTER, TEMPO);
}

interface MidiEvent {
  tick: number;
  type: 'noteOn' | 'noteOff';
  note: number;
  velocity: number;
}

function encodeMidi(events: MidiEvent[], ticksPerQuarter: number, tempo: number): Uint8Array {
  const chunks: number[] = [];
  
  // Header chunk (MThd)
  chunks.push(...stringToBytes('MThd'));
  chunks.push(...int32ToBytes(6));
  chunks.push(...int16ToBytes(0)); // Format 0
  chunks.push(...int16ToBytes(1)); // One track
  chunks.push(...int16ToBytes(ticksPerQuarter));
  
  // Track chunk (MTrk)
  const trackData: number[] = [];
  
  // Tempo meta event
  trackData.push(0); // Delta time
  trackData.push(0xFF); // Meta event
  trackData.push(0x51); // Tempo
  trackData.push(0x03); // Length
  trackData.push((tempo >> 16) & 0xFF);
  trackData.push((tempo >> 8) & 0xFF);
  trackData.push(tempo & 0xFF);
  
  // Note events
  let lastTick = 0;
  for (const event of events) {
    const deltaTick = event.tick - lastTick;
    trackData.push(...encodeVariableLength(deltaTick));
    
    if (event.type === 'noteOn') {
      trackData.push(0x90); // Note On, channel 0
      trackData.push(event.note);
      trackData.push(event.velocity);
    } else {
      trackData.push(0x80); // Note Off, channel 0
      trackData.push(event.note);
      trackData.push(event.velocity);
    }
    
    lastTick = event.tick;
  }
  
  // End of track
  trackData.push(0);
  trackData.push(0xFF);
  trackData.push(0x2F);
  trackData.push(0x00);
  
  // Track header
  chunks.push(...stringToBytes('MTrk'));
  chunks.push(...int32ToBytes(trackData.length));
  chunks.push(...trackData);
  
  return new Uint8Array(chunks);
}

function encodeVariableLength(value: number): number[] {
  const bytes: number[] = [];
  let buffer = value & 0x7F;
  
  while ((value >>= 7) > 0) {
    buffer <<= 8;
    buffer |= 0x80;
    buffer += (value & 0x7F);
  }
  
  while (true) {
    bytes.push(buffer & 0xFF);
    if (buffer & 0x80) {
      buffer >>= 8;
    } else {
      break;
    }
  }
  
  return bytes;
}

function stringToBytes(str: string): number[] {
  return str.split('').map(c => c.charCodeAt(0));
}

function int32ToBytes(value: number): number[] {
  return [
    (value >> 24) & 0xFF,
    (value >> 16) & 0xFF,
    (value >> 8) & 0xFF,
    value & 0xFF
  ];
}

function int16ToBytes(value: number): number[] {
  return [
    (value >> 8) & 0xFF,
    value & 0xFF
  ];
}
