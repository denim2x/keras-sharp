using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KerasSharp.Utils {
  static class Collections {
    public static IEnumerable<TSource> skip<TSource>(this ICollection<TSource> source, int count) {
      if (count < 0) {
        count += source.Count;
      }
      return source.Skip(count);
    }

    public static TSource get<TSource>(this List<TSource> source, int index) {
      if (index < 0) {
        index += source.Count;
      }
      return source[index];
    }

    public static TSource get<TSource>(this TSource[] source, int index) {
      if (index < 0) {
        index += source.Length;
      }
      return source[index];
    }

    public static TSource[] Concatenate<TSource>(this TSource[] source, TSource[] other) {
      return source.Concat(other).ToArray();
    }
  }
}
